%* *******************
*** SHARED PROGRAM ***
*********************%
#program shared.

#const n=10.
node(1..n).

%* ******************
*** GUESS PROGRAM ***
*********************
Specifies candidate solutions:
- connected undirected graphs with a 3-coloring of their leaves
- (satisfying some mild conditions to rule out uninteresting cases)
***************** *%
#program guess.

% Guess an undirected graph with n nodes
{ edge(X,Y) : node(X), node(Y), X < Y }.
edge(X,Y) :- edge(Y,X).

% Ensure that it is connected
reachable(1).
reachable(N) :- reachable(M), edge(N,M).
:- node(N), not reachable(N).

% Define a coloring on the leaves
leaf(N) :- node(N), edge(N,M1),
    not edge(N,M2) : node(M2), M1 != M2.
color(1..3).
1 { leaf_color(N,C) : color(C) } 1 :- leaf(N).

% No node may have two or more leaves with the same color as neighbors
:- node(N), edge(N,M1), edge(N,M2), M1 < M2,
    leaf_color(M1,C), leaf_color(M2,C).
% No node may have three or more leaves as neighbors
:- node(N), #count { M : edge(N,M), leaf(M) } >= 3.

% Show the edges (without duplicates)
#show.
#show edge(X,Y) : edge(X,Y), X < Y.
% Show the leaves and the coloring
#show leaf/1.
#show leaf_color/2.


%* *****************
*** GLUE PROGRAM ***
********************
Specifies the atoms on which the guess and check programs interact.
***************** *%
#program glue.

% Guess and check interact on nodes and edges
edge(X,Y) :- node(X), node(Y), X != Y.

% And on leaves and their coloring
leaf(1..n).
color(1..3).
leaf_color(N,C) :- node(N), color(C).


%* ******************
*** CHECK PROGRAM ***
*********************
Specifies counterexamples that rule out candidate solutions:
- a proper 3-coloring of the entire graph that agrees with the leaf coloring
***************** *%
#program check.

% Check that there is no proper 3-coloring of the entire graph
% that agrees with the leaf coloring
1 { color(N,C) : color(C) } 1 :- node(N).
:- edge(N,M), color(N,C), color(M,C).
:- leaf_color(N,C), not color(N,C), leaf(N).
:- color(N,C), not leaf_color(N,C), leaf(N).

% Any counterexample found depends only on these Stm for which relevant(Stm)
relevant(edge(N,M)) :- node(N), node(M), N < M, not edge(N,M).
relevant(leaf_color(N,C)) :- leaf_color(N,C), leaf(N).
