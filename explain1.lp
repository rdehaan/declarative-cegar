% https://staff.fnwi.uva.nl/u.endriss/pubs/files/BoixelEndrissAAMAS2020.pdf
% https://staff.fnwi.uva.nl/u.endriss/pubs/files/NardiEtAlAAMAS2022.pdf

%* *** TODO's ***
- finish comments
- 'Borda' constraints on choices, also for plurality, etc.
- neutrality/3
*************** *%

%* *******************
*** SHARED PROGRAM ***
*********************%
#program shared.

%%% The voting/justification setting
% #const num_candidates=3.
candidate(1..num_candidates).
position(1..num_candidates).

% #const num_voters=3.
voter(1..num_voters).

% #const num_profiles=3.
profile_num(1..num_profiles).

% #const num_axiom_instances=3.

% Auxiliary
swap(C1,C2,C1,C2) :-
    candidate(C1), candidate(C2), C1 < C2.
swap(C1,C2,C2,C1) :-
    candidate(C1), candidate(C2), C1 < C2.
swap(C1,C2,C3,C3) :-
    candidate(C1), candidate(C2), C1 < C2,
    candidate(C3), C3 != C1, C3 != C2.

%%% Declare axioms
% #external use_axiom(pareto).
% #external use_axiom(faithfulness).
% #external use_axiom(cancellation).
% #external use_axiom(reinforcement).
% #external use_axiom(condorcet).
% #external use_axiom(neutrality2).
% #external use_axiom(pos_responsiveness).

% use_axiom(pareto).
% use_axiom(faithfulness).
% use_axiom(cancellation).
% use_axiom(reinforcement).
% use_axiom(condorcet).
% use_axiom(neutrality2).
% use_axiom(pos_responsiveness).

axiom(pareto(C)) :- candidate(C),
    use_axiom(pareto).
pos_axiom_instance(single(Prof),pareto(C)) :-
    use_axiom(pareto),
    profile_num(Prof), axiom(pareto(C)).

axiom(faithfulness) :-
    use_axiom(faithfulness).
pos_axiom_instance(single(Prof),faithfulness) :-
    use_axiom(faithfulness),
    profile_num(Prof), axiom(faithfulness).

axiom(unanimity) :-
    use_axiom(unanimity).
pos_axiom_instance(single(Prof),unanimity) :-
    use_axiom(unanimity),
    profile_num(Prof), axiom(unanimity).

axiom(cancellation) :-
    use_axiom(cancellation).
pos_axiom_instance(single(Prof),cancellation) :-
    use_axiom(cancellation),
    profile_num(Prof), axiom(cancellation).

axiom(reinforcement) :-
    use_axiom(reinforcement).
reinforcement_triple(Prof1,Prof2,Prof3) :-
    use_axiom(reinforcement),
    profile_num(Prof1), profile_num(Prof2), Prof1 <= Prof2,
    profile_num(Prof3),
    Prof1 < Prof3, Prof2 < Prof3. % only allowed due to symmetry breaking
reinforcement_triple(Prof1,Prof2,1) :-
    use_axiom(reinforcement),
    profile_num(Prof1), profile_num(Prof2), Prof1 <= Prof2.
pos_axiom_instance(triple(Prof1,Prof2,Prof3),reinforcement) :-
    use_axiom(reinforcement),
    reinforcement_triple(Prof1,Prof2,Prof3),
    axiom(reinforcement).

axiom(condorcet(C)) :- candidate(C),
    use_axiom(condorcet).
pos_axiom_instance(single(Prof),condorcet(C)) :-
    use_axiom(condorcet),
    profile_num(Prof), axiom(condorcet(C)).

axiom(neutrality2(C1,C2)) :- candidate(C1), candidate(C2), C1 < C2,
    use_axiom(neutrality2).
pos_axiom_instance(double(Prof1,Prof2),neutrality2(C1,C2)) :-
    use_axiom(neutrality2),
    profile_num(Prof1), profile_num(Prof2), Prof1 <= Prof2,
    axiom(neutrality2(C1,C2)).

pos_response_pair(C,P) :-
    use_axiom(pos_responsiveness),
    candidate(C), position(P), P < num_candidates.
axiom(pos_responsiveness(V,C,P)) :-
    use_axiom(pos_responsiveness),
    voter(V), pos_response_pair(C,P).
pos_axiom_instance(double(Prof1,Prof2),pos_responsiveness(V,C,P)) :-
    use_axiom(pos_responsiveness),
    profile_num(Prof1), profile_num(Prof2),
    axiom(pos_responsiveness(V,C,P)).

%* *******************************
*** Target profile and outcome ***
*** CAREFUL: put the profile   ***
*** in lexicographic order :-) ***
******************************* *%

% use_axiom(pareto).
% #const num_candidates=3.
% #const num_voters=2.
% #const num_profiles=1.
% #const num_axiom_instances=2.
% target_profile(1,1,1).
% target_profile(1,2,2).
% target_profile(1,3,3).
% target_profile(2,1,1).
% target_profile(2,3,2).
% target_profile(2,2,3).
% target_outcome(1).
% requires_polarity_axiom.

% use_axiom(condorcet).
% #const num_candidates=3.
% #const num_voters=2.
% #const num_profiles=1.
% #const num_axiom_instances=1.
% target_profile(1,1,1).
% target_profile(1,2,2).
% target_profile(1,3,3).
% target_profile(2,1,1).
% target_profile(2,3,2).
% target_profile(2,2,3).
% target_outcome(1).
% requires_polarity_axiom.

% use_axiom(faithfulness).
% use_axiom(reinforcement).
% #const num_candidates=3.
% #const num_voters=2.
% #const num_profiles=3.
% #const num_axiom_instances=3.
% target_profile(1,1,1).
% target_profile(1,2,2).
% target_profile(1,3,3).
% target_profile(2,1,1).
% target_profile(2,3,2).
% target_profile(2,2,3).
% target_outcome(1).
% requires_polarity_axiom.

% use_axiom(unanimity).
% #const num_candidates=3.
% #const num_voters=2.
% #const num_profiles=1.
% #const num_axiom_instances=1.
% target_profile(1,1,1).
% target_profile(1,2,2).
% target_profile(1,3,3).
% target_profile(2,1,1).
% target_profile(2,3,2).
% target_profile(2,2,3).
% target_outcome(1).
% requires_polarity_axiom.

% use_axiom(faithfulness).
% #const num_candidates=3.
% #const num_voters=1.
% #const num_profiles=1.
% #const num_axiom_instances=1.
% target_profile(1,1,1).
% target_profile(1,2,2).
% target_profile(1,3,3).
% target_outcome(1).
% requires_polarity_axiom.

% use_axiom(pareto).
% #const num_candidates=3.
% #const num_voters=1.
% #const num_profiles=1.
% #const num_axiom_instances=2.
% target_profile(1,1,1).
% target_profile(1,2,2).
% target_profile(1,3,3).
% target_outcome(1).
% requires_polarity_axiom.

% use_axiom(cancellation).
% #const num_candidates=3.
% #const num_voters=2.
% #const num_profiles=1.
% #const num_axiom_instances=1.
% target_profile(1,1,1).
% target_profile(1,2,2).
% target_profile(1,3,3).
% target_profile(2,3,1).
% target_profile(2,2,2).
% target_profile(2,1,3).
% target_outcome(1).
% target_outcome(2).
% target_outcome(3).

% use_axiom(faithfulness).
% use_axiom(reinforcement).
% #const num_candidates=3.
% #const num_voters=2.
% #const num_profiles=2.
% #const num_axiom_instances=2.
% target_profile(1,1,1).
% target_profile(1,2,2).
% target_profile(1,3,3).
% target_profile(2,1,1).
% target_profile(2,2,2).
% target_profile(2,3,3).
% target_outcome(1).
% requires_polarity_axiom.

% use_axiom(faithfulness).
% use_axiom(cancellation).
% use_axiom(reinforcement).
% #const num_candidates=3.
% #const num_voters=3.
% #const num_profiles=3.
% #const num_axiom_instances=3.
% target_profile(1,1,1).
% target_profile(1,2,2).
% target_profile(1,3,3).
% target_profile(2,1,1).
% target_profile(2,2,2).
% target_profile(2,3,3).
% target_profile(3,3,1).
% target_profile(3,2,2).
% target_profile(3,1,3).
% target_outcome(1).
% requires_polarity_axiom.

% use_axiom(pareto).
% use_axiom(neutrality2).
% #const num_candidates=3.
% #const num_voters=2.
% #const num_profiles=1.
% #const num_axiom_instances=2.
% target_profile(1,1,1).
% target_profile(1,2,2).
% target_profile(1,3,3).
% target_profile(2,2,1).
% target_profile(2,1,2).
% target_profile(2,3,3).
% target_outcome(1).
% target_outcome(2).
% requires_polarity_axiom.

% use_axiom(faithfulness).
% use_axiom(cancellation).
% use_axiom(reinforcement).
% use_axiom(pareto).
% #const num_candidates=3.
% #const num_voters=3.
% #const num_profiles=6.
% #const num_axiom_instances=3.
% target_profile(1,1,1).
% target_profile(1,2,2).
% target_profile(1,3,3).
% target_profile(2,1,1).
% target_profile(2,2,2).
% target_profile(2,3,3).
% target_profile(3,3,1).
% target_profile(3,2,2).
% target_profile(3,1,3).
% target_outcome(1).
% requires_polarity_axiom.

% % UNSAT
% use_axiom(faithfulness).
% use_axiom(cancellation).
% use_axiom(reinforcement).
% use_axiom(pareto).
% use_axiom(condorcet).
% #const num_candidates=3.
% #const num_voters=3.
% #const num_profiles=5.
% #const num_axiom_instances=3.
% target_profile(1,1,1).
% target_profile(1,2,2).
% target_profile(1,3,3).
% target_profile(2,1,1).
% target_profile(2,2,2).
% target_profile(2,3,3).
% target_profile(3,3,1).
% target_profile(3,2,2).
% target_profile(3,1,3).
% target_outcome(2).
% requires_polarity_axiom.

% use_axiom(pareto).
% use_axiom(neutrality2).
% use_axiom(reinforcement).
% #const num_candidates=3.
% #const num_voters=3.
% #const num_profiles=3.
% #const num_axiom_instances=5.
% target_profile(1,1,1).
% target_profile(1,2,2).
% target_profile(1,3,3).
% target_profile(2,1,1).
% target_profile(2,3,2).
% target_profile(2,2,3).
% target_profile(3,2,1).
% target_profile(3,1,2).
% target_profile(3,3,3).
% target_outcome(1).
% requires_polarity_axiom.

% use_axiom(faithfulness).
% use_axiom(pareto).
% use_axiom(neutrality2).
% use_axiom(reinforcement).
% #const num_candidates=3.
% #const num_voters=3.
% #const num_profiles=3.
% #const num_axiom_instances=4.
% target_profile(1,1,1).
% target_profile(1,2,2).
% target_profile(1,3,3).
% target_profile(2,1,1).
% target_profile(2,3,2).
% target_profile(2,2,3).
% target_profile(3,2,1).
% target_profile(3,1,2).
% target_profile(3,3,3).
% target_outcome(1).
% requires_polarity_axiom.

% use_axiom(cancellation).
% use_axiom(pos_responsiveness).
% #const num_candidates=3.
% #const num_voters=2.
% #const num_profiles=2.
% #const num_axiom_instances=2.
% target_profile(1,1,1).
% target_profile(1,2,2).
% target_profile(1,3,3).
% target_profile(2,2,1).
% target_profile(2,3,2).
% target_profile(2,1,3).
% target_outcome(2).
% requires_polarity_axiom.

use_axiom(cancellation).
use_axiom(reinforcement).
use_axiom(pos_responsiveness).
#const num_candidates=3.
#const num_voters=4.
#const num_profiles=4.
#const num_axiom_instances=4.
target_profile(1,1,1).
target_profile(1,2,2).
target_profile(1,3,3).
target_profile(2,1,1).
target_profile(2,3,2).
target_profile(2,2,3).
target_profile(3,2,1).
target_profile(3,3,2).
target_profile(3,1,3).
target_profile(4,3,1).
target_profile(4,1,2).
target_profile(4,2,3).
target_outcome(1).
requires_polarity_axiom.

% % FOR EFFICIENCY, NEEDS 2-STEP PROCEDURE
% % (FIRST FINDING PROFILES)
% use_axiom(pareto).
% use_axiom(neutrality2).
% use_axiom(cancellation).
% use_axiom(reinforcement).
% #const num_candidates=3.
% #const num_voters=5.
% #const num_profiles=6.
% #const num_axiom_instances=8.
% target_profile(1,1,1).
% target_profile(1,2,2).
% target_profile(1,3,3).
% target_profile(2,3,1).
% target_profile(2,1,2).
% target_profile(2,2,3).
% target_profile(3,3,1).
% target_profile(3,1,2).
% target_profile(3,2,3).
% target_outcome(1).
% target_outcome(3).
% requires_polarity_axiom.

%* ******************
*** GUESS PROGRAM ***
*********************
Specifies candidate solutions:
- TODO
****************** *%
#program guess.

%%% Guess profiles
{ active_voter(Prof,V) } :- profile_num(Prof), voter(V).
:- active_voter(Prof,V), not active_voter(Prof,V-1), voter(V), voter(V-1).
active_profile(Prof) :- profile_num(Prof), active_voter(Prof,1).
:- active_profile(Prof), profile_num(Prof-1), not active_profile(Prof-1).

1 { profile(Prof,V,Cand,Pos) : position(Pos) } 1 :-
    profile_num(Prof), voter(V), active_voter(Prof,V), candidate(Cand).
1 { profile(Prof,V,Cand,Pos) : candidate(Cand) } 1 :-
    profile_num(Prof), voter(V), active_voter(Prof,V), position(Pos).

%%% Symmetry breaking: lexicographical ordering within each profile
lex_eq_until(Prof,V1,V2,0) :-
    profile_num(Prof), voter(V1), voter(V2), V1 < V2.
lex_eq_until(Prof,V1,V2,Pos) :-
    profile_num(Prof), voter(V1), voter(V2), V1 < V2,
    position(Pos), lex_eq_until(Prof,V1,V2,Pos-1),
    profile(Prof,V1,C,Pos), profile(Prof,V2,C,Pos).
lex_dom(Prof,V1,V2) :- profile_num(Prof), voter(V1), voter(V2), V1 < V2,
    profile(Prof,V1,C1,Pos), profile(Prof,V2,C2,Pos), C2 < C1,
    lex_eq_until(Prof,V1,V2,Pos-1).
:- profile_num(Prof), lex_dom(Prof,V1,V2), voter(V1), voter(V2), V1 < V2,
    active_voter(Prof,V1), active_voter(Prof,V2).

%%% Make sure that the target profile appears with index 1
:- not profile(1,V,Cand,Pos), target_profile(V,Cand,Pos).
target_active_voter(V) :- target_profile(V,_,_).
:- active_voter(1,V), not target_active_voter(V).

%%% Shared for applicability of various axioms
num_active_voters(Prof,Num) :-
    profile_num(Prof), Num = #count { V : active_voter(Prof,V) }.
votes_identical(Prof1,V1,Prof2,V2) :-
    profile_num(Prof1), profile_num(Prof2),
    active_voter(Prof1,V1), active_voter(Prof2,V2),
    profile(Prof1,V1,C,P) : profile(Prof2,V2,C,P).
num_votes_identical(Prof1,V1,Prof2,N) :-
    profile_num(Prof1), profile_num(Prof2), voter(V1),
    N = #count { V2 : voter(V2), votes_identical(Prof1,V1,Prof2,V2) }.

%%% Applicability of PARETO
dominates_in_voter(Prof,V,C1,C2) :-
    use_axiom(pareto),
    profile_num(Prof), active_voter(Prof,V),
    candidate(C1), candidate(C2), C1 != C2,
    profile(Prof,V,C1,P1), profile(Prof,V,C2,P2), P1 < P2.
dominates_in_profile(Prof,C1,C2) :-
    use_axiom(pareto),
    profile_num(Prof),
    candidate(C1), candidate(C2), C1 != C2,
    dominates_in_voter(Prof,V,C1,C2) : active_voter(Prof,V).
axiom_applicable(single(Prof),pareto(C2)) :-
    use_axiom(pareto),
    pos_axiom_instance(single(Prof),pareto(C2)),
    active_profile(Prof),
    dominates_in_profile(Prof,C1,C2), candidate(C1), C1 != C2.

%%% Applicability of FAITHFULNESS
axiom_applicable(single(Prof),faithfulness) :-
    use_axiom(faithfulness),
    pos_axiom_instance(single(Prof),faithfulness),
    active_profile(Prof),
    active_voter(Prof,1), not active_voter(Prof,2).

%%% Applicability of UNANIMITY
axiom_applicable(single(Prof),unanimity) :-
    use_axiom(unanimity),
    pos_axiom_instance(single(Prof),unanimity),
    active_profile(Prof), candidate(C),
    profile(Prof,V,C,1) : active_voter(Prof,V).

%%% Applicability of CANCELLATION
num_active_voters_even(Prof) :-
    use_axiom(cancellation),
    num_active_voters(Prof,Num), Num \ 2 == 0.
cancellation_flipped_pair(Prof,V1,V2) :-
    use_axiom(cancellation),
    num_active_voters_even(Prof), num_active_voters(Prof,Num),
    active_voter(Prof,V1), active_voter(Prof,V2), V1 < V2,
    V1 <= Num/2, V2 == Num-V1+1.
cancellation_identical_pair(Prof,V1,V2) :-
    use_axiom(cancellation),
    num_active_voters_even(Prof), num_active_voters(Prof,Num),
    active_voter(Prof,V1), active_voter(Prof,V2), V1 < V2,
    V1 <= Num/2, V2 <= Num/2.
cancellation_identical_pair(Prof,V1,V2) :-
    use_axiom(cancellation),
    num_active_voters_even(Prof), num_active_voters(Prof,Num),
    active_voter(Prof,V1), active_voter(Prof,V2), V1 < V2,
    V1 > Num/2, V2 > Num/2.
votes_identical(Prof,V1,V2) :-
    votes_identical(Prof,V1,Prof,V2), V1 < V2.
votes_flipped(Prof,V1,V2) :-
    use_axiom(cancellation),
    profile_num(Prof), active_voter(Prof,V1), active_voter(Prof,V2), V1 < V2,
    profile(Prof,V2,C,num_candidates-P+1) : profile(Prof,V1,C,P).
axiom_applicable(single(Prof),cancellation) :-
    use_axiom(cancellation),
    pos_axiom_instance(single(Prof),cancellation),
    active_profile(Prof),
    num_active_voters_even(Prof),
    votes_identical(Prof,V1,V2) : cancellation_identical_pair(Prof,V1,V2);
    votes_flipped(Prof,V1,V2) : cancellation_flipped_pair(Prof,V1,V2).

%%% Applicability of REINFORCEMENT
reinforcement_count_check(triple(Prof1,Prof2,Prof3),V) :-
    use_axiom(reinforcement),
    pos_axiom_instance(triple(Prof1,Prof2,Prof3),reinforcement),
    active_voter(Prof3,V),
    num_votes_identical(Prof3,V,Prof1,N1),
    num_votes_identical(Prof3,V,Prof2,N2),
    num_votes_identical(Prof3,V,Prof3,N3),
    N3 = N1 + N2.
axiom_applicable(triple(Prof1,Prof2,Prof3),reinforcement) :-
    use_axiom(reinforcement),
    pos_axiom_instance(triple(Prof1,Prof2,Prof3),reinforcement),
    active_profile(Prof1), active_profile(Prof2), active_profile(Prof3),
    num_active_voters(Prof1,N1), num_active_voters(Prof2,N2),
    num_active_voters(Prof3,N3), N3 = N1 + N2,
    reinforcement_count_check(triple(Prof1,Prof2,Prof3),V) :
        active_voter(Prof3,V).

%%% Applicability of CONDORCET
beats(Prof,V,C1,C2) :-
    use_axiom(condorcet),
    profile_num(Prof), active_voter(Prof,V),
    candidate(C1), candidate(C2), C1 != C2,
    profile(Prof,V,C1,P1), profile(Prof,V,C2,P2), P1 < P2.
majority_beats(Prof,C1,C2) :-
    use_axiom(condorcet),
    profile_num(Prof), candidate(C1), candidate(C2), C1 != C2,
    N1 = #count { V : beats(Prof,V,C1,C2) },
    N2 = #count { V : beats(Prof,V,C2,C1) },
    N1 > N2.
axiom_applicable(single(Prof),condorcet(C1)) :-
    use_axiom(condorcet),
    pos_axiom_instance(single(Prof),condorcet(C1)),
    active_profile(Prof),
    majority_beats(Prof,C1,C2) : candidate(C2), C1 != C2.

%%% Applicability of NEUTRALITY/2
votes_identical_under_swap(Prof1,V1,Prof2,V2,C1,C2) :-
    use_axiom(neutrality2),
    profile_num(Prof1), profile_num(Prof2),
    active_voter(Prof1,V1), active_voter(Prof2,V2),
    candidate(C1), candidate(C2), C1 < C2,
    profile(Prof1,V1,CS2,P) : profile(Prof2,V2,CS1,P), swap(C1,C2,CS1,CS2).
num_votes_identical_under_swap(Prof1,V1,Prof2,C1,C2,N) :-
    use_axiom(neutrality2),
    profile_num(Prof1), profile_num(Prof2), voter(V1),
    candidate(C1), candidate(C2), C1 < C2,
    N = #count { V2 : voter(V2),
                      votes_identical_under_swap(Prof1,V1,Prof2,V2,C1,C2) }.
neutrality2_count_check(double(Prof1,Prof2),C1,C2,V) :-
    use_axiom(neutrality2),
    pos_axiom_instance(double(Prof1,Prof2),neutrality2(C1,C2)),
    active_voter(Prof1,V),
    num_votes_identical(Prof1,V,Prof1,N),
    num_votes_identical_under_swap(Prof1,V,Prof2,C1,C2,N).
axiom_applicable(double(Prof1,Prof2),neutrality2(C1,C2)) :-
    use_axiom(neutrality2),
    pos_axiom_instance(double(Prof1,Prof2),neutrality2(C1,C2)),
    active_profile(Prof1), active_profile(Prof2),
    num_active_voters(Prof1,N), num_active_voters(Prof2,N),
    neutrality2_count_check(double(Prof1,Prof2),C1,C2,V) :
        active_voter(Prof1,V).

%%% Applicability of POSITIVE RESPONSIVENESS
votes_identical_after_upwards_move(Prof1,V1,Prof2,V2,C,P) :-
    use_axiom(pos_responsiveness),
    profile_num(Prof1), profile_num(Prof2),
    active_voter(Prof1,V1), active_voter(Prof2,V2),
    pos_response_pair(C,P),
    profile(Prof1,V1,C,P1), profile(Prof2,V2,C,P1-P),
    profile(Prof2,V2,C2,P2) :
        profile(Prof1,V1,C2,P2), P2 > P1;
    profile(Prof2,V2,C2,P2) :
        profile(Prof1,V1,C2,P2), P2 < P1-P;
    profile(Prof2,V2,C2,P2+1) :
        profile(Prof1,V1,C2,P2), P2 < P1, P2 >= P1-P.
pos_response_check(Prof1,V1,Prof2,V2,C,P) :-
    use_axiom(pos_responsiveness),
    profile_num(Prof1), profile_num(Prof2),
    active_voter(Prof1,V1), active_voter(Prof2,V2),
    pos_response_pair(C,P),
    votes_identical_after_upwards_move(Prof1,V1,Prof2,V2,C,P),
    num_votes_identical(Prof2,V2,Prof2,N2),
    num_votes_identical(Prof2,V2,Prof1,N1),
    N2 = N1 + 1.
pos_response_check(Prof1,V1,Prof2,V2,C,P) :-
    use_axiom(pos_responsiveness),
    profile_num(Prof1), profile_num(Prof2),
    active_voter(Prof1,V1), active_voter(Prof2,V2),
    pos_response_pair(C,P),
    votes_identical(Prof1,V1,Prof2,V2),
    num_votes_identical(Prof2,V2,Prof2,N2),
    num_votes_identical(Prof2,V2,Prof1,N1),
    N2 = N1 - 1.
pos_response_check(Prof1,V1,Prof2,V2,C,P) :-
    use_axiom(pos_responsiveness),
    profile_num(Prof1), profile_num(Prof2),
    active_voter(Prof1,V1), active_voter(Prof2,V2),
    pos_response_pair(C,P),
    not votes_identical_after_upwards_move(Prof1,V1,Prof2,V2,C,P),
    not votes_identical(Prof1,V1,Prof2,V2),
    num_votes_identical(Prof2,V2,Prof2,N2),
    num_votes_identical(Prof2,V2,Prof1,N1),
    N2 = N1.
axiom_applicable(double(Prof1,Prof2),pos_responsiveness(V,C,P)) :-
    use_axiom(pos_responsiveness),
    pos_axiom_instance(double(Prof1,Prof2),pos_responsiveness(V,C,P)),
    active_profile(Prof1), active_profile(Prof2),
    num_active_voters(Prof1,N), num_active_voters(Prof2,N),
    pos_response_check(Prof1,V,Prof2,V2,C,P) : active_voter(Prof2,V2).

%%% Guess the right amount of axiom instances to use
num_axiom_instances { use_axiom_instance(Profiles,Axiom) :
    axiom_applicable(Profiles,Axiom) } num_axiom_instances.
% use_axiom_instance(Profiles,Axiom) :-
%     axiom_applicable(Profiles,Axiom).

%%% Keep track of which axioms have instances used
is_used(pareto) :- use_axiom_instance(Pr,pareto(C)).
is_used(unanimity) :- use_axiom_instance(Pr,unanimity).
is_used(faithfulness) :- use_axiom_instance(Pr,faithfulness).
is_used(cancellation) :- use_axiom_instance(Pr,cancellation).
is_used(reinforcement) :- use_axiom_instance(Pr,reinforcement).
is_used(condorcet) :- use_axiom_instance(Pr,condorcet(C)).
is_used(neutrality2) :- use_axiom_instance(Pr,neutrality2(C1,C2)).
is_used(pos_responsiveness) :- use_axiom_instance(Pr,pos_responsiveness(V,C,P)).

polarity_axiom_used :- is_used(pareto).
polarity_axiom_used :- is_used(unanimity).
polarity_axiom_used :- is_used(faithfulness).
polarity_axiom_used :- is_used(condorcet).
polarity_axiom_used :- is_used(pos_responsiveness).
:- requires_polarity_axiom, not polarity_axiom_used.

%%% Redundant constraint: only use subsequent profile numbers
:- profile_num(Prof), profile_num(Prof+1),
    profile_used(Prof+1), not profile_used(Prof).

%%% Symmetry breaking: only guess profiles that are actually used
profile_used(Prof) :- use_axiom_instance(single(Prof),_).
profile_used(Prof) :- use_axiom_instance(double(Prof,_),_).
profile_used(Prof) :- use_axiom_instance(double(_,Prof),_).
profile_used(Prof) :- use_axiom_instance(triple(Prof,_,_),_).
profile_used(Prof) :- use_axiom_instance(triple(_,Prof,_),_).
profile_used(Prof) :- use_axiom_instance(triple(_,_,Prof),_).
:- active_profile(Prof), not profile_used(Prof).

%%% Symmetry breaking: only use profiles that are connected to the target
profile_connected(1).
profile_connected(Prof1) :-
    profile_connected(Prof2),
    use_axiom_instance(double(Prof1,Prof2),_).
profile_connected(Prof1) :-
    profile_connected(Prof2),
    use_axiom_instance(double(Prof2,Prof1),_).
profile_connected(Prof1) :-
    profile_connected(Prof2),
    use_axiom_instance(triple(Prof1,Prof2,_),_).
profile_connected(Prof1) :-
    profile_connected(Prof2),
    use_axiom_instance(triple(Prof1,_,Prof2),_).
profile_connected(Prof1) :-
    profile_connected(Prof2),
    use_axiom_instance(triple(_,Prof1,Prof2),_).
profile_connected(Prof1) :-
    profile_connected(Prof2),
    use_axiom_instance(triple(Prof2,Prof1,_),_).
profile_connected(Prof1) :-
    profile_connected(Prof2),
    use_axiom_instance(triple(Prof2,_,Prof1),_).
profile_connected(Prof1) :-
    profile_connected(Prof2),
    use_axiom_instance(triple(_,Prof2,Prof1),_).
:- profile_used(Prof), not profile_connected(Prof).

% %%% Redundant constraint: provide upper bound on number of profiles to use
% %%% (SEEMS NOT TO GIVE A GOOD TRADEOFF)
% max_num_profiles(S) :-
%     S2 = #sum { 1,Prof1,Prof2,Axiom :
%         use_axiom_instance(double(Prof1,Prof2),Axiom) },
%     S3 = #sum { 2,Prof1,Prof2,Prof3,Axiom :
%         use_axiom_instance(triple(Prof1,Prof2,Prof3),Axiom) },
%     S = 1 + S2 + S3.
% :- max_num_profiles(S), profile_used(S+1).

%%% Symmetry breaking: ensure that profiles (except no 1)
%%% are sorted by size
:- profile_used(Prof1), profile_used(Prof2), Prof1 < Prof2,
    Prof1 != 1, Prof2 != 1,
    num_active_voters(Prof1,N1),
    num_active_voters(Prof2,N2),
    N2 < N1.

%%% Symmetry breaking: ensure that same-size profiles (except no 1)
%%% are lexicographically sorted
interprof_vote_lex_eq_until(Prof1,Prof2,V,0) :-
    profile_num(Prof1), profile_num(Prof2), Prof1 < Prof2, voter(V).
interprof_vote_lex_eq_until(Prof1,Prof2,V,Pos) :-
    profile_num(Prof1), profile_num(Prof2), Prof1 < Prof2, voter(V),
    position(Pos), interprof_vote_lex_eq_until(Prof1,Prof2,V,Pos-1),
    profile(Prof1,V,C,Pos), profile(Prof2,V,C,Pos).
interprof_vote_lex_dom(Prof1,Prof2,V) :-
    profile_num(Prof1), profile_num(Prof2), Prof1 < Prof2, voter(V),
    profile(Prof1,V,C1,Pos), profile(Prof2,V,C2,Pos), C2 < C1,
    interprof_vote_lex_eq_until(Prof1,Prof2,V,Pos-1).
interprof_lex_eq_until(Prof1,Prof2,0) :-
    profile_num(Prof1), profile_num(Prof2), Prof1 < Prof2.
interprof_lex_eq_until(Prof1,Prof2,V) :-
    profile_num(Prof1), profile_num(Prof2), Prof1 < Prof2, voter(V),
    interprof_lex_eq_until(Prof1,Prof2,V-1),
    votes_identical(Prof1,V,Prof2,V).
interprof_lex_dom(Prof1,Prof2) :-
    profile_num(Prof1), profile_num(Prof2), Prof1 < Prof2, voter(V),
    interprof_lex_eq_until(Prof1,Prof2,V-1),
    interprof_vote_lex_dom(Prof1,Prof2,V).
:- profile_num(Prof1), profile_num(Prof2), Prof1 < Prof2,
    num_active_voters(Prof1,N), num_active_voters(Prof2,N),
    interprof_lex_dom(Prof2,Prof1).


#show profile/4.
#show use_axiom_instance/2.


%* *****************
*** GLUE PROGRAM ***
********************
Specifies the atoms on which the guess and check programs interact.
***************** *%
#program glue.

% Profiles
glue(profile(Prof,V,Cand,Pos)) :-
    profile_num(Prof), voter(V), candidate(Cand), position(Pos).

% Axiom instances used
glue(use_axiom_instance(Profiles,Axiom)) :- pos_axiom_instance(Profiles,Axiom).

% glue(axiom_applicable(Profiles,Axiom) :- pos_axiom_instance(Profiles,Axiom).

%* ******************
*** CHECK PROGRAM ***
*********************
Specifies counterexamples for candidate solutions:
- TODO
****************** *%
#program check.

%%% Guess (irresolute) outcomes for all profiles
1 { outcome(Prof,C) : candidate(C) } :- profile_num(Prof).

%%% Make sure that the outcome for profile 1 differs from the target outcome
target_outcome_differs :-
    outcome(1,C), not target_outcome(C), candidate(C).
target_outcome_differs :-
    not outcome(1,C), target_outcome(C), candidate(C).
:- not target_outcome_differs.

%%% Heuristic: satisfy as many axioms as possible
#heuristic falsified(pos_axiom_instance(Profiles,Axiom)) :
    use_axiom_instance(Profiles,Axiom). [20,false]
#heuristic falsified(pos_axiom_instance(Profiles,Axiom)) :
    pos_axiom_instance(Profiles,Axiom). [10,false]

%%% Make sure that the outcomes satisfy all used axiom instances
:- use_axiom_instance(Profiles,Axiom),
    falsified(pos_axiom_instance(Profiles,Axiom)).
%%% PARETO
falsified(pos_axiom_instance(single(Prof),pareto(C))) :-
    pos_axiom_instance(single(Prof),pareto(C)),
    outcome(Prof,C).
%%% FAITHFULNESS
falsified(pos_axiom_instance(single(Prof),faithfulness)) :-
    pos_axiom_instance(single(Prof),faithfulness),
    profile(Prof,1,C,1), not outcome(Prof,C).
falsified(pos_axiom_instance(single(Prof),faithfulness)) :-
    pos_axiom_instance(single(Prof),faithfulness),
    profile(Prof,1,C1,1),
    outcome(Prof,C2), candidate(C2), C1 != C2.
%%% UNANIMITY
falsified(pos_axiom_instance(single(Prof),unanimity)) :-
    pos_axiom_instance(single(Prof),unanimity),
    profile(Prof,1,C,1), not outcome(Prof,C).
falsified(pos_axiom_instance(single(Prof),unanimity)) :-
    pos_axiom_instance(single(Prof),unanimity),
    profile(Prof,1,C1,1),
    outcome(Prof,C2), candidate(C2), C1 != C2.
%%% CANCELLATION
falsified(pos_axiom_instance(single(Prof),cancellation)) :-
    pos_axiom_instance(single(Prof),cancellation),
    not outcome(Prof,C), candidate(C).
%%% REINFORCEMENT
outcomes_intersect(Prof1,Prof2) :-
    profile_num(Prof1), profile_num(Prof2), Prof1 <= Prof2,
    outcome(Prof1,C), outcome(Prof2,C), candidate(C).
falsified(pos_axiom_instance(triple(Prof1,Prof2,Prof3),reinforcement)) :-
    pos_axiom_instance(triple(Prof1,Prof2,Prof3),reinforcement),
    outcomes_intersect(Prof1,Prof2),
    outcome(Prof3,C), not outcome(Prof1,C).
falsified(pos_axiom_instance(triple(Prof1,Prof2,Prof3),reinforcement)) :-
    pos_axiom_instance(triple(Prof1,Prof2,Prof3),reinforcement),
    outcomes_intersect(Prof1,Prof2),
    outcome(Prof3,C), not outcome(Prof2,C).
falsified(pos_axiom_instance(triple(Prof1,Prof2,Prof3),reinforcement)) :-
    pos_axiom_instance(triple(Prof1,Prof2,Prof3),reinforcement),
    outcomes_intersect(Prof1,Prof2),
    outcome(Prof1,C), outcome(Prof2,C), not outcome(Prof3,C).
%%% CONDORCET
falsified(pos_axiom_instance(single(Prof),condorcet(C))) :-
    pos_axiom_instance(single(Prof),condorcet(C)),
    not outcome(Prof,C).
falsified(pos_axiom_instance(single(Prof),condorcet(C1))) :-
    pos_axiom_instance(single(Prof),condorcet(C1)),
    outcome(Prof,C2), candidate(C2), C1 != C2.
%%% NEUTRALITY/2
falsified(pos_axiom_instance(double(Prof1,Prof2),neutrality2(C1,C2))) :-
    pos_axiom_instance(double(Prof1,Prof2),neutrality2(C1,C2)),
    swap(C1,C2,CS1,CS2), candidate(CS1),
    outcome(Prof1,CS1), not outcome(Prof2,CS2).
falsified(pos_axiom_instance(double(Prof1,Prof2),neutrality2(C1,C2))) :-
    pos_axiom_instance(double(Prof1,Prof2),neutrality2(C1,C2)),
    swap(C1,C2,CS1,CS2), candidate(CS1),
    outcome(Prof2,CS2), not outcome(Prof1,CS1).
%%% POSITIVE RESPONSIVENESS
falsified(pos_axiom_instance(double(Prof1,Prof2),pos_responsiveness(V,C,P))) :-
    pos_axiom_instance(double(Prof1,Prof2),pos_responsiveness(V,C,P)),
    candidate(C),
    outcome(Prof1,C), not outcome(Prof2,C).
falsified(pos_axiom_instance(double(Prof1,Prof2),pos_responsiveness(V,C1,P))) :-
    pos_axiom_instance(double(Prof1,Prof2),pos_responsiveness(V,C1,P)),
    candidate(C1), candidate(C2), C1 != C2,
    outcome(Prof1,C1), outcome(Prof2,C2).


% %%% REFINEMENT STRATEGY 1
% relevant(profile(Prof,V,Cand,Pos)) :-
%     profile_num(Prof), voter(V), candidate(Cand), position(Pos).
% relevant(use_axiom_instance(Profiles,Axiom)) :-
%     falsified(pos_axiom_instance(Profiles,Axiom)),
%     pos_axiom_instance(Profiles,Axiom).

%%% REFINEMENT STRATEGY 2
relevant(profile(Prof,V,Cand,Pos)) :-
    profile_num(Prof), voter(V), candidate(Cand), position(Pos).
relevant(use_axiom_instance(Profiles,Axiom)) :-
    not falsified(pos_axiom_instance(Profiles,Axiom)),
    pos_axiom_instance(Profiles,Axiom).

% %%% REFINEMENT STRATEGY 3
% relevant(profile(Prof,V,Cand,Pos)) :-
%     profile_num(Prof), voter(V), candidate(Cand), position(Pos).
% relevant(use_axiom_instance(Profiles,Axiom)) :-
%     not use_axiom_instance(Profiles,Axiom),
%     pos_axiom_instance(Profiles,Axiom).

% %%% REFINEMENT STRATEGY 4
% relevant(profile(Prof,V,Cand,Pos)) :-
%     profile_num(Prof), voter(V), candidate(Cand), position(Pos).
% relevant(use_axiom_instance(Profiles,Axiom)) :-
%     use_axiom_instance(Profiles,Axiom).

% %%% REFINEMENT STRATEGY 5
% relevant(profile(Prof,V,Cand,Pos)) :-
%     profile_num(Prof), voter(V), candidate(Cand), position(Pos).
% relevant(use_axiom_instance(Profiles,Axiom)) :-
%     pos_axiom_instance(Profiles,Axiom).
