#!/bin/bash

# This script (multi_condition/run_tdnn_7k.sh) is the reverberated version of
# tuning/run_tdnn_7k.sh. It reverberates the training data with room impulse responses
# which leads to better results.
# (The reverberation of data is done in multi_condition/run_ivector_common.sh)
# This script assumes a mixing of the original training data with its reverberated copy
# and results in a 2-fold training set. Thus the number of epochs is halved to
# keep the same training time. The model converges after 2 epochs of training,
# The WER doesn't change much with more epochs of training.
# local/chain/compare_wer_general.sh tdnn_7k_sp/ tdnn_7k_sp_rvb1/
# System                tdnn_7k_sp/ tdnn_7k_sp_rvb1/
# WER on train_dev(tg)      13.81     13.91
# WER on train_dev(fg)      12.67     12.86
# WER on eval2000(tg)        16.3      16.1
# WER on eval2000(fg)        14.7      14.3
# Final train prob         -0.087    -0.122
# Final valid prob         -0.109    -0.130
# Final train prob (xent)        -1.269    -1.561
# Final valid prob (xent)       -1.3184   -1.5727


set -e

# configs for 'chain'
affix=
multi=multi_a
stage=1
test_online_decoding=true
train_stage=-10
get_egs_stage=-10
speed_perturb=false
dir=exp/$multi/chain/tdnn_lstm_7k_max1_r631_d800  # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=
decode_nj=30
ivector_dir=exp/$multi/nnet3_rvb_r631
num_data_reps=2        # number of reverberated copies of data to generate
input_train_set=$multi/tdnn

# training options
num_epochs=2
initial_effective_lrate=0.001
final_effective_lrate=0.0001
leftmost_questions_truncate=-1
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=4
num_jobs_final=4
minibatch_size=128
frames_per_eg=150
remove_egs=false
common_egs_dir=
xent_regularize=0.1
has_fisher=false

#lstm spec stuff
label_delay=5

chunk_left_context=40
chunk_right_context=0
# we'll put chunk-left-context-initial=0 and chunk-right-context-final=0
# directly without variables.
frames_per_chunk=140,100,160

# (non-looped) decoding options
frames_per_chunk_primary=$(echo $frames_per_chunk | cut -d, -f1)
extra_left_context=50
extra_right_context=0

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

dir=${dir}${affix:+_$affix}${suffix}_rvb${num_data_reps}
clean_train_set=${input_train_set}${suffix}
train_set=${clean_train_set}_rvb${num_data_reps}
ali_dir=exp/$multi/tri5_ali
treedir=exp/chain/tri5_7k_tree$suffix
lang=data/lang_chain_7k
clean_lat_dir=exp/tri5_lats_nodup${suffix}
lat_dir=${clean_lat_dir}_rvb${num_data_reps}


# The data reverberation will be done in this script.
local/nnet3/multi_condition/run_ivector_common.sh --stage $stage \
  --input-data-dir ${input_train_set} \
  --ivector-dir $ivector_dir \
  --speed-perturb $speed_perturb \
  --num-data-reps $num_data_reps || exit 1;

#stage=12 # skip to the model

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  #nj=$(cat exp/tri4_ali_nodup${suffix}/num_jobs) || exit 1;
  #steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/${clean_train_set} \
  #  data/lang exp/tri4 $clean_lat_dir
  #rm $clean_lat_dir/fsts.*.gz # save space


  # Create the lattices for the reverberated data
  # We use the lattices/alignments from the clean data for the reverberated data.
  mkdir -p $lat_dir/temp/
  lattice-copy "ark:gunzip -c $clean_lat_dir/lat.*.gz |" ark,scp:$lat_dir/temp/lats.ark,$lat_dir/temp/lats.scp

  # copy the lattices for the reverberated data
  rm -f $lat_dir/temp/combined_lats.scp
  touch $lat_dir/temp/combined_lats.scp
  # Here prefix "rev0_" represents the clean set, "rev1_" represents the reverberated set
  for i in `seq 0 $num_data_reps`; do
    cat $lat_dir/temp/lats.scp | sed -e "s/^/rev${i}_/" >> $lat_dir/temp/combined_lats.scp
  done
  sort -u $lat_dir/temp/combined_lats.scp > $lat_dir/temp/combined_lats_sorted.scp

  lattice-copy scp:$lat_dir/temp/combined_lats_sorted.scp "ark:|gzip -c >$lat_dir/lat.1.gz" || exit 1;
  echo "1" > $lat_dir/num_jobs

  # copy other files from original lattice dir
  for f in cmvn_opts final.mdl splice_opts tree; do
    cp $clean_lat_dir/$f $lat_dir/$f
  done
fi


if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  # we build the tree using the clean alignments as we empirically found that this was better.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 7000 data/${clean_train_set} $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  [ -z $num_targets ] && { echo "$0: error getting num-targets"; exit 1; }
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  lstm_opts="decay-time=20"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=800
  relu-renorm-layer name=tdnn2 input=Append(-1,0,1) dim=800
  relu-renorm-layer name=tdnn3 input=Append(-1,0,1) dim=800

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  fast-lstmp-layer name=fastlstm1 cell-dim=800 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $lstm_opts
  relu-renorm-layer name=tdnn4 input=Append(-3,0,3) dim=800
  relu-renorm-layer name=tdnn5 input=Append(-3,0,3) dim=800
  fast-lstmp-layer name=fastlstm2 cell-dim=800 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $lstm_opts
  relu-renorm-layer name=tdnn6 input=Append(-3,0,3) dim=800
  relu-renorm-layer name=tdnn7 input=Append(-3,0,3) dim=800
  fast-lstmp-layer name=fastlstm3 cell-dim=800 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $lstm_opts

  ## adding the layers for chain branch
  output-layer name=output input=fastlstm3 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=fastlstm3 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/

fi



if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-reverb-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  elif [[ $(hostname -f) == Goliath ]]; then
    utils/create_split_dir.pl /media/ssd/speechtmp/multi_en-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage \
                              /media/ssd2/speechtmp/multi_en-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage \
    			      /media/drive3/speechtmp/multi_en-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage \
                              $dir/egs/storage
    #echo "skipping here"
  fi


steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir ${ivector_dir}/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.num-chunk-per-minibatch 64,32 \
    --trainer.frames-per-iter 1500000 \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs 2 \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial 4 \
    --trainer.optimization.num-jobs-final 4 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.optimization.momentum 0.0 \
    --trainer.deriv-truncate-margin 8 \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_chunk \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;

fi

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_fsh_sw1_tg $dir $dir/graph_fsh_sw1_tg
fi

graph_dir=$dir/graph_fsh_sw1_tg
if [ $stage -le 15 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in eval2000; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir $ivector_dir/ivectors_${decode_set} \
          $graph_dir data/${decode_set}_hires/test \
          $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_tg || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;
      fi
      )
  done
fi

if $test_online_decoding && [ $stage -le 17 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
       --mfcc-config conf/mfcc_hires.conf \
       $lang $ivector_dir/extractor $dir ${dir}_online

  rm $dir/.error 2>/dev/null || true
  for decode_set in eval2000; do
    (
      # note: we just give it "$decode_set" as it only uses the wav.scp, the
      # feature type does not matter.

      steps/online/nnet3/decode.sh --nj $decode_nj --cmd "$decode_cmd" $iter_opts \
          --acwt 1.0 --post-decode-acwt 10.0 \
         $graph_dir data/${decode_set}_hires/test \
         ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_tg || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;
      fi
    ) || touch $dir/.error #&
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in online decoding"
    exit 1
  fi
fi


wait;
exit 0;
