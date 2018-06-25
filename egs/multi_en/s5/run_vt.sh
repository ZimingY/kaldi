#!/bin/bash

stage=8
multi=multi_a

. ./cmd.sh
. ./path.sh 

set -e

export CUDA_VISIBLE_DEVICES=1

kwd_set=heyolly

if  [ $stage -le 1 ]; then
  #kwd/prepare_kwds_aws.sh
  mkdir -p data/$kwd_set
  utils/data/subset_data_dir.sh --first data/local/kwds_all 22000 data/$kwd_set/train
  utils/data/subset_data_dir.sh --last data/local/kwds_all  1821 data/$kwd_set/test
  #for f in wav.scp utt2spk spk2utt utt2dur text; do
  #   [ ! -f $f ] && echo "Expected $f to exist. Exiciting" && exit 1;
  #   cp data/local/$kwd_set/$f data/$kwd_set/$f
  #done
fi

if [ $stage -le 2 ]; then
  mfccdir=mfcc
  corpora="$kwd_set"

  for c in $corpora; do
    data=data/$c/train
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf \
      --cmd "$train_cmd" --nj 15 \
      $data exp/make_mfcc/$c/train || exit 1;
    steps/compute_cmvn_stats.sh \
      $data exp/make_mfcc/$c/train || exit 1;
    utils/fix_data_dir.sh $data 
  done

  for c in $corpora; do
    data=data/$c/test
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf \
      --cmd "$train_cmd" --nj 5 \
      $data exp/make_mfcc/$c/test || exit 1;
    steps/compute_cmvn_stats.sh \
      $data exp/make_mfcc/$c/test || exit 1;
    utils/fix_data_dir.sh $data
  done

fi

if [ $stage -le 3 ]; then
  corpora="$kwd_set"
  for c in $corpora; do
    data=data/$c/train
    steps/align_fmllr.sh --cmd "$train_cmd" --nj 30 --careful true \
      $data data/lang \
      exp/$multi/tri5 exp/$multi/tri5_ali_$kwd_set || exit 1;
  done

  for c in $corpora; do
    data=data/$c/test
    steps/align_fmllr.sh --cmd "$train_cmd" --nj 10 --careful true \
      $data data/lang \
      exp/$multi/tri5 exp/$multi/tri5_ali_${kwd_set}_test || exit 1;
  done

fi


if [ $stage -le 4 ]; then

  #local/make_partitions.sh --multi $multi --stage 8 || exit 1;
  #make targets for this model
  steps/get_train_ctm.sh --print-silence true --use-segments false data/multi_a/tri5_ali data/lang exp/multi_a/tri5_ali
  steps/get_train_ctm.sh --print-silence true --use-segments false data/heyolly/train/ data/lang exp/multi_a/tri5_ali_heyolly

fi

ddir=data/multi_a/tdnn_30k_vt
edir=exp/multi_a/tdnn_30k_vt

if [ $stage -le 5 ]; then

  #ddir=data/multi_a/tdnn_30k_vt
  #edir=exp/multi_a/tdnn_30k_vt

  mkdir -p $edir

  kwrd=kwd/word2class.list
  cat exp/multi_a/tri5_ali/ctm exp/multi_a/tri5_ali_heyolly/ctm | sort -k1,1 |\
    kwd/convert_ctm_to_vt_ids.pl $kwrd $ddir | sort -k1 |\
      copy-int-vector ark,t:- ark,scp:$edir/targets.ark,$edir/targets.scp || exit 1;

fi


if [ $stage -le 6 ]; then
  num_data_repos=1

  mkdir -p $edir/temp/
  #copy-feats scp:$edir/targets.scp ark,scp:$edir/temp/targets.ark,$edir/temp/targets.scp

  # copy the lattices for the reverberated data
  rm -f $edir/temp/combined_ali.scp
  touch $edir/temp/combined_ali.scp
  # Here prefix "rev0_" represents the clean set, "rev1_" represents the reverberated set
  for i in `seq 0 $num_data_repos`; do
    cat $edir/targets.scp | sed -e "s/^/rev${i}_/" >> $edir/temp/combined_ali.scp
  done
  sort -u $edir/temp/combined_ali.scp > $edir/temp/combined_ali_sorted.scp

  copy-int-vector scp:$edir/temp/combined_ali_sorted.scp ark,scp:$edir/targets_rvb$num_data_repos.ark,$edir/targets_rvb$num_data_repos.scp || exit 1;
  cat $edir/targets_rvb$num_data_repos.scp | sort -k1 > $edir/targets_all.scp
  rm -r $edir/temp
fi

#exit;

if [ $stage -le 7 ]; then
  kwd/run_tdnn_vt.sh --stage 9 --train-stage 0
fi

#large scale one, only 2m utts from multi_a due to disk constraints on whole asr data
if [ $stage -le 8 ]; then
  kwd/run_tdnn_vt_asr.sh --stage 10 --train-stage 0
fi
