"""
Example guess and check based solver for second level problems.
(based on https://github.com/potassco/clingo/tree/master/examples/clingo/gac)
"""

import sys
from typing import Sequence, Tuple, List

from clingo import ast
from clingo.ast import ProgramBuilder
from clingo.control import Control
from clingo.application import clingo_main, Application
from clingo.propagator import PropagateControl, PropagateInit, Propagator
from clingo.backend import Backend
from clingo.ast import parse_files, AST, ASTType
from clingo.symbol import Function


class Transformer:
    """
    Transformer for the guess and check solver.
    """
    _builder: ProgramBuilder
    _state: str
    _check: List[AST]
    _glue: List[AST]

    def __init__(
        self,
        builder: ProgramBuilder,
        check: List[AST],
        glue: List[str]
    ):
        self._builder = builder
        self._state = "guess"
        self._check = check
        self._glue = glue

    def add(self, stm: AST):
        """
        Add the given statement to the guess or check programs.
        """
        if stm.ast_type == ASTType.Program:
            if stm.name == "shared" and not stm.parameters:
                self._state = "shared"
            elif stm.name == "glue" and not stm.parameters:
                self._state = "glue"
            elif stm.name == "check" and not stm.parameters:
                self._state = "check"
            elif (stm.name in ["base", "guess"]) and not stm.parameters:
                self._state = "guess"
            else:
                raise RuntimeError("unexpected program part")

        else:
            if self._state == "shared":
                self._builder.add(stm)
                self._check.append(stm)
                self._glue.append(stm)
            elif self._state == "guess":
                self._builder.add(stm)
            elif self._state == "check":
                self._check.append(stm)
            elif self._state == "glue":
                self._glue.append(stm)


class Checker:
    """
    Class wrapping a solver to perform the second level check.
    """
    _ctl: Control
    _map: List[Tuple[int, int, int]]

    def __init__(self):
        self._ctl = Control(["--heuristic=Domain"])
        self._map = []

    def backend(self) -> Backend:
        """
        Return the backend of the underlying solver.
        """
        return self._ctl.backend()

    def add(self, guess_lit: int, check_lit: int, indicator_lit: int):
        """
        Map the given solver literal to the corresponding program literal in
        the checker.
        """
        self._map.append((guess_lit, check_lit, indicator_lit))

    def ground(self, check: Sequence[ast.AST]):
        """
        Ground the check program.
        """
        with ProgramBuilder(self._ctl) as bld:
            for stm in check:
                bld.add(stm)

        self._ctl.ground([("base", [])])

    def check(self, control: PropagateControl) -> bool:
        """
        Return true if the check program is unsatisfiable w.r.t. to the atoms
        of the guess program.

        The truth values of the atoms of the guess program are stored in the
        assignment of the given control object.
        """

        assignment = control.assignment

        assumptions = []
        for guess_lit, check_lit, _ in self._map:
            guess_truth = assignment.value(guess_lit)
            assumptions.append(check_lit if guess_truth else -check_lit)

        with self._ctl.solve(assumptions, yield_=True) as handle:
            result = handle.get()
            model = handle.model()
            mask = []
            if model:
                for guess_lit, check_lit, indicator_lit in self._map:
                    if model.is_true(indicator_lit):
                        mask.append(guess_lit)
            if result.unsatisfiable is not None:
                return result.unsatisfiable, mask

        raise RuntimeError("search interrupted")


class CheckPropagator(Propagator):
    """
    Simple propagator verifying that a check program holds on total
    assignments.
    """
    _check: List[AST]
    _glue: List[str]
    _checkers: List[Checker]
    _gluelits: List[int]

    def __init__(self, check: List[AST], glue: List[AST]):
        self._check = check
        self._glue = glue
        self._checkers = []
        self._gluelits = []

    def init(self, init: PropagateInit):
        """
        Initialize the solvers for the check programs.
        """
        # we need a checker for each thread (to be able to solve in parallel)
        for _ in range(init.number_of_threads):
            checker = Checker()
            self._checkers.append(checker)

            with checker.backend() as backend:
                for atom in init.symbolic_atoms:

                    # ignore atoms that are not glue
                    if str(atom.symbol) not in self._glue:
                        continue

                    guess_lit = init.solver_literal(atom.literal)
                    guess_truth = init.assignment.value(guess_lit)

                    # ignore false atoms
                    if guess_truth is False:
                        continue

                    check_lit = backend.add_atom(atom.symbol)

                    indicator_symbol = Function("relevant", [atom.symbol], True)
                    indicator_lit = backend.add_atom(indicator_symbol)

                    if guess_lit not in self._gluelits:
                        self._gluelits.append(guess_lit)

                    # fix true atoms
                    if guess_truth is True:
                        backend.add_rule([check_lit], [])

                    # add a choice rule for unknow atoms and add them to the
                    # mapping table of the checker
                    else:
                        backend.add_rule([check_lit], [], True)
                        backend.add_rule([indicator_lit], [], True)
                        checker.add(guess_lit, check_lit, indicator_lit)

            checker.ground(self._check)


    def check(self, control: PropagateControl):
        """
        Check total assignments.
        """
        assignment = control.assignment
        checker = self._checkers[control.thread_id]

        unsatisfiable, mask = checker.check(control)
        if not unsatisfiable:

            conflict = []

            for lit in self._gluelits:
                if lit in mask:
                    conflict.append(-lit if assignment.is_true(lit) else lit)

            control.add_clause(conflict)


class GCCApp(Application):
    """
    Application class implementing a custom solver.
    """
    program_name: str
    version: str

    def __init__(self):
        self.program_name = "declarative-cegar"
        self.version = "0.1"

    def main(self, control: Control, files: Sequence[str]):
        """
        The main function called with a Control object and a list of files
        passed on the command line.
        """
        if not files:
            files = ["-"]

        check: List[AST] = []
        glue: List[AST] = []
        with ProgramBuilder(control) as bld:
            trans = Transformer(bld, check, glue)
            parse_files(files, trans.add)

        # Take glue program, find a model, and collects its shown atoms as str
        glue_control = Control()
        with ProgramBuilder(glue_control) as glue_bld:
            for stm in glue:
                glue_bld.add(stm)
        glue_control.ground([("base", [])])
        glue_control.configuration.solve.models = 1
        glue = []
        def on_model(model):
            for atom in model.symbols(shown=True):
                glue.append(str(atom))
        glue_control.solve(on_model=on_model)

        control.register_propagator(CheckPropagator(check, glue))

        control.ground([("base", [])])
        control.solve()


sys.exit(clingo_main(GCCApp(), sys.argv[1:]))
