#!/bin/bash

RAPID_MATCH="./build/matching/RapidMatch.out -order nd -preprocess true -num MAX -time_limit 600"

graphs=( wiki-vote epinions amazon dblp google youtube patents wiki-talk )
queries=( query-1 query-2 query-3 query-4 query-5 query-6 query-7 )

for g in "${graphs[@]}"
do
  for q in "${queries[@]}"
  do
    $RAPID_MATCH -d ../graph_match/data/$g.graph -q ../graph_match/data/$q.graph
  done
done

snb_graphs=( sf1 sf3 sf10 sf60 )
snb_queries=( snb1 snb3 snb4 snb6 snb7 snb8 snb9 )

for g in "${snb_graphs[@]}"
do
  for q in "${snb_queries[@]}"
  do
    $RAPID_MATCH -d ../graph_match/data/$g.graph -q ../graph_match/data/$q.graph
  done
done
