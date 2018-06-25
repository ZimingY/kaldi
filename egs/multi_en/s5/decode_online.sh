#/bin/bash

mdir=exp/multi_a/chain
mdls="tdnn_7k_max1_rn_d800_rvb2"
dsets="devboardsimtest1__-r_16000_remix_1"
#dsets=devboardsimtest1__-r_16000_remix_1
#dsets="test_wav_data_834__-r_16000_remix_1"

perutts="true"
suffix=_ma7k_b14_lb5_sw0md15
#lang_suffix=amt50k.fsh_sw1.int09
lang_suffix=amt64knorm.fsh_pr1e-7.int088
#lang_suffix=fsh_sw1_tg
mkgraph=true
endpoint=false
silence_weight=0.0
max_state_duration=15

#--endpoint.rule1.min-trailing-silence=0.5
#--endpoint.rule1.min-utterance-length=1.5
#--endpoint.rule2.min-trailing-silence=0.05
#--endpoint.rule3.min-trailing-silence=0.1
#--endpoint.rule4.min-trailing-silence=0.2
#--endpoint.rule4.min-utterance-length=1.5

for perutt in $perutts; do

  suffixt=$suffix
  [ "$perutt" == "true" ] && suffixt=${suffix}_perutt

  for mdl in $mdls; do

    if $mkgraph; then
      utils/mkgraph.sh --self-loop-scale 1.0 data/lang_$lang_suffix $mdir/$mdl $mdir/$mdl/graph_$lang_suffix 
    fi

    for dset in $dsets; do
      nj=`cat testsets/$dset/spk2utt | wc -l`
      steps/online/nnet3/decode.sh --nj $nj --max-active 7000 --beam 14.0 --lattice-beam 5.0 \
        --per-utt $perutt --acwt 1.0 --post-decode-acwt 10.0 --do-endpointing $endpoint \
        --silence-weight $silence_weight --max-state-duration $max_state_duration \
        $mdir/$mdl/graph_$lang_suffix testsets/$dset $mdir/${mdl}_online/decode_${dset}_${lang_suffix}${suffixt} &
    done

    wait;

    for dset in $dsets; do
      grep WER ${mdir}/${mdl}_online/decode_${dset}_${lang_suffix}${suffixt}/wer*
    done

  done

done
