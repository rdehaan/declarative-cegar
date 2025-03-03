%* *******************
*** SHARED PROGRAM ***
*********************%
#program shared.

#const n=5.
player(1..n).

#const k=4.
color(1..4).

#const w=3.
weight(1..w).

%* ******************
*** GUESS PROGRAM ***
********************%
#program guess.

% Guess an undirected graph
{ edge(X,Y) : player(X), player(Y), X < Y }.
edge(X,Y) :- edge(Y,X).

% Ensure that the graph is connected
reachable(1).
reachable(N) :- reachable(M), edge(N,M).
:- player(N), not reachable(N).

% Ensure that the graph is not a tree
path_without(N,N,M) :- player(N), player(M), N != M.
path_without(N1,N3,M) :- path_without(N1,N2,M),
    edge(N2,N3), N3 != M.
cycle :- path_without(N1,N2,M), edge(N1,M), edge(N2,M).
:- not cycle.

%%% Symmetry breaking on undirected graphs
% from: https://doi.org/10.1007/s10601-018-9294-5
sb_max_index(n).
sb_index(1..M) :- sb_max_index(M).
sb_pair(I,J) :- sb_index(I), sb_index(J), I < J.

sb_seq_length(I,J,M-2) :- sb_pair(I,J), sb_max_index(M).
sb_triple(I,J,1..L) :- sb_pair(I,J), sb_seq_length(I,J,L).
sb_counter(I,J,K,U) :- sb_triple(I,J,K), U = K, I > K.
sb_counter(I,J,K,U) :- sb_triple(I,J,K), U = K+1, I <= K, J > K+1.
sb_counter(I,J,K,U) :- sb_triple(I,J,K), U = K+2, I = K, J = K+1.
sb_counter(I,J,K,U) :- sb_triple(I,J,K), U = K+2, J <= K+1.
sb_statement(I,J,K,first,edge(I,U)) :- sb_counter(I,J,K,U).
sb_statement(I,J,K,second,edge(J,U)) :- sb_counter(I,J,K,U).

sb_y(I,J,0) :- sb_pair(I,J).
:- sb_triple(I,J,K),
    sb_statement(I,J,K,first,edge(F1,F2)),
    sb_statement(I,J,K,second,edge(S1,S2)),
    sb_y(I,J,K-1), edge(F1,F2), not edge(S1,S2).
sb_y(I,J,K) :- sb_triple(I,J,K), K < L, sb_seq_length(I,J,L),
    sb_statement(I,J,K,first,edge(F1,F2)),
    sb_y(I,J,K-1), edge(F1,F2).
sb_y(I,J,K) :- sb_triple(I,J,K), K < L, sb_seq_length(I,J,L),
    sb_statement(I,J,K,second,edge(S1,S2)),
    sb_y(I,J,K-1), not edge(S1,S2).

% Guess color sets of players
1 { color_available(P,C) : color(C) } :- player(P).

% Guess weights on edges
1 { edge(X,Y,W) : weight(W) } 1 :- edge(X,Y).
edge(X,Y,W) :- edge(Y,X,W).

% Guess Nash equilibrium
1 { plays(P,C) : color_available(P,C) } 1 :- player(P).
score(P,C,S) :- player(P), color_available(P,C),
    S = #sum { W,Q : edge(P,Q,W), plays(Q,C) }.
:- score(P,C1,S1), score(P,C2,S2),
    color(C1), color(C2), C1 != C2,
    plays(P,C1), S2 > S1.

% Optional: rule out 'trivial' case where all players play the same color
:- color(C), plays(P,C) : player(P).

% Optional: require that cumulative size of color sets is large
%:- not #count { P,C : color_available(P,C) } >= 10.

% Optional: heuristics
#heuristic edge(P,Q) : player(P), player(Q). [10,true]
#heuristic color_available(P,C) : player(P), color(C). [20,true]

#show.
#show edge(X,Y,W) : edge(X,Y,W), X < Y.
#show color_available/2.
#show plays/2.


%* *****************
*** GLUE PROGRAM ***
*******************%
#program glue.

edge(P,Q,W) :- player(P), player(Q), weight(W).
color_available(P,C) :- player(P), color(C).
plays(P,C) :- player(P), color(C).


%* ******************
*** CHECK PROGRAM ***
********************%
#program check.

% Guess a(nother) Nash equilibrium
1 { plays_alt(P,C) : color_available(P,C) } 1 :- player(P).
score_alt(P,C,S) :- player(P), color_available(P,C),
    S = #sum { W,Q : edge(P,Q,W), plays_alt(Q,C) }.
:- score_alt(P,C1,S1), score_alt(P,C2,S2),
    color(C1), color(C2), C1 != C2,
    plays_alt(P,C1), S2 > S1.

% Try to make the difference small (for stronger refinement using strategy 2)
#heuristic plays_alt(P,C) : plays(P,C). [10,true]

% Check that the two equilibria are different
player_differs(P) :- player(P), plays(P,C1), plays_alt(P,C2), C1 != C2.
equilibrium_differs :- player_differs(P).
:- not equilibrium_differs.

%%% Refinement strategy 1:
%%% All glue atoms are relevant (trivial refinement)
% relevant(edge(P,Q,W)) :- player(P), player(Q), weight(W).
% relevant(color_available(P,C)) :- player(P), color(C).
% relevant(plays(P,C)) :- player(P), color(C).

%%% Refinement strategy 2:
%%% Only 'neighborhood' of difference is relevant
% relevant(edge(P,Q,W)) :- player(P), player(Q), weight(W), player_differs(P).
% relevant(color_available(P,C)) :- player(P), color(C), player_differs(P).
% adjacent_to_differing_player(P) :- player(P),
%     player(Q), edge(P,Q,_), player_differs(Q).
% relevant(plays(P,C)) :- player(P), color(C), adjacent_to_differing_player(P).

%%% Refinement strategy 3:
%%% Only 'neighborhood' of difference is relevant
%%% (stricter definition of neighborhood)
% - either an edge between a differing player and another player has to change
relevant(edge(P,Q,W)) :- player(P), player(Q), weight(W), player_differs(P).
% - or a color set has to change for a differing player
% (i. on what they play in one of the two equilibria)
relevant(color_available(P,C)) :- player(P), color(C), player_differs(P),
    plays(P,C).
relevant(color_available(P,C)) :- player(P), color(C), player_differs(P),
    plays_alt(P,C).
% (ii. or on what they don't play, but what a neighbor plays)
relevant(color_available(P,C)) :- player(P), color(C), player_differs(P),
    not color_available(P,C), player(Q), edge(P,Q), plays(Q,C).
relevant(color_available(P,C)) :- player(P), color(C), player_differs(P),
    not color_available(P,C), player(Q), edge(P,Q), plays_alt(Q,C).
% - or the played move (in the guessed equilibrium)
% for a non-differing player has to change
adjacent_to_differing_player(P) :- player(P),
    player(Q), edge(P,Q,_), player_differs(Q).
relevant(plays(P,C)) :- player(P), color(C), adjacent_to_differing_player(P).
