#!/bin/bash

#this does basic data augmentation on heyolly data, this one is latter combined with typical asr data to train  VT on asr-data

. ./path.sh

stage=15
exp=heyolly
speed_perturb=true
affix=sp
multi=multi_a
num_data_reps=2
dir=exp/$multi/$exp
dir=$dir${affix:+_$affix}
train_set=$multi/${exp}

kwd/run_common.sh --stage $stage --multi $multi \
  --input-data-dir ${train_set} \
  --speed-perturb $speed_perturb \
  --num-data-reps $num_data_reps || exit 1;

if [ $stage -le 9 ]; then

  #for f in final.mdl tree; do
  #  cp exp/$multi/tri5/$f $dir/$f
  #done

  #local/make_partitions.sh --multi $multi --stage 8 || exit 1;
  #make targets for this model
  #steps/get_train_ctm.sh --print-silence true --use-segments false data/multi_a/tri5_ali data/lang exp/multi_a/tri5_ali
  steps/get_train_ctm.sh --print-silence true --use-segments false data/${train_set}_sp data/lang  exp/$multi/tri5_ali_${exp}_sp
  cp exp/$multi/tri5_ali_${exp}_sp/ctm $dir/ctm
fi

if [ $stage -le 10 ]; then

  #ddir=data/multi_a/tdnn_30k_vt
  #edir=exp/multi_a/tdnn_30k_vt

  #for f in final.mdl tree; do
  #  cp exp/$multi/tri5/$f $dir/$f
  #done

  #mkdir -p $dir

  kwrd=kwd/word2class.list
  cat exp/multi_a/tri5_ali_${exp}_sp/ctm | sort -k1,1 |\
    kwd/convert_ctm_to_vt_ids.pl $kwrd data/${train_set}_sp | sort -k1 |\
      copy-int-vector ark,t:- ark,scp:$dir/targets.ark,$dir/targets.scp || exit 1;

fi


if [ $stage -le 11 ]; then
 
  edir=$dir

  mkdir -p $edir/temp/
  #copy-feats scp:$edir/targets.scp ark,scp:$edir/temp/targets.ark,$edir/temp/targets.scp

  # copy the lattices for the reverberated data
  rm -f $edir/temp/combined_ali.scp
  touch $edir/temp/combined_ali.scp
  # Here prefix "rev0_" represents the clean set, "rev1_" represents the reverberated set
  for i in `seq 0 $num_data_reps`; do
    cat $edir/targets.scp | sed -e "s/^/rev${i}_/" >> $edir/temp/combined_ali.scp
  done
  sort -u $edir/temp/combined_ali.scp > $edir/temp/combined_ali_sorted.scp

  copy-int-vector scp:$edir/temp/combined_ali_sorted.scp ark,scp:$edir/targets_rvb$num_data_reps.ark,$edir/targets_rvb$num_data_reps.scp || exit 1;
  cat $edir/targets_rvb$num_data_reps.scp | sort -k1 > $edir/targets_all.scp
  #rm -r $edir/temp

fi


if [ $stage -le 13 ]; then

  #for f in final.mdl tree; do
  #  cp exp/$multi/tri5/$f $dir/$f
  #done

  #local/make_partitions.sh --multi $multi --stage 8 || exit 1;
  #make targets for this model
  steps/get_train_ctm.sh --print-silence true --use-segments false data/multi_a/tri5_ali data/lang exp/multi_a/tri5_ali
  #steps/get_train_ctm.sh --print-silence true --use-segments false data/${train_set}_sp data/lang  exp/$multi/tri5_ali_${exp}_sp
fi

if [ $stage -le 14 ]; then

  #ddir=data/multi_a/tdnn_30k_vt
  #edir=exp/multi_a/tdnn_30k_vt

  #for f in final.mdl tree; do
  #  cp exp/$multi/tri5/$f $dir/$f
  #done

  #mkdir -p $dir

  kwrd=kwd/word2class.list
  cat exp/multi_a/tri5_ali/ctm | sort -k1,1 |\
    kwd/convert_ctm_to_vt_ids.pl $kwrd data/$multi/tri5_ali | sort -k1 |\
      copy-int-vector ark,t:- ark,scp:exp/$multi/tri5_ali/targets.ark,exp/$multi/tri5_ali/targets.scp || exit 1;

fi



if [ $stage -le 15 ]; then
  num_data_reps=2
  dir=exp/multi_a/ 
  edir=exp/$multi/tri5_ali/

  mkdir -p $edir/temp/
  #copy-feats scp:$edir/targets.scp ark,scp:$edir/temp/targets.ark,$edir/temp/targets.scp

  # copy the lattices for the reverberated data
  rm -f $edir/temp/combined_ali.scp
  touch $edir/temp/combined_ali.scp
  # Here prefix "rev0_" represents the clean set, "rev1_" represents the reverberated set
  for i in `seq 0 $num_data_reps`; do
    cat $edir/targets.scp | sed -e "s/^/rev${i}_/" >> $edir/temp/combined_ali.scp
  done
  sort -u $edir/temp/combined_ali.scp > $edir/temp/combined_ali_sorted.scp

  copy-int-vector scp:$edir/temp/combined_ali_sorted.scp ark,scp:$edir/targets_rvb$num_data_reps.ark,$edir/targets_rvb$num_data_reps.scp || exit 1;
  cat $edir/targets_rvb$num_data_reps.scp | sort -k1 > $edir/targets_all.scp
  #rm -r $edir/temp

fi

