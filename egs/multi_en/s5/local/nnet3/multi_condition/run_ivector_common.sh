#!/bin/bash 
#set -e
# This script is based on local/nnet3/run_ivector_common.sh.
# It reverberates the original data with simulated room impulse responses

. ./cmd.sh

stage=1
multi=multi_a
num_data_reps=1  # number of reverberated copies of data to generate
                 # These will be combined with the original data.
input_data_dir=$multi/tdnn
ivector_dir=exp/$multi/nnet3_rvb_ratios_631
speed_perturb=false
estimate_ivec=true

set -e
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

mkdir -p $ivector_dir

# Here we recommend speed perturbation as the gains are significant.
# The gain from speed perturbation is additive with the gain from data reverberation
if [ "$speed_perturb" == "true" ]; then
  # perturbed data preparation
  if [ $stage -le 1 ] && [ ! -f data/${input_data_dir}_sp/feats.scp ]; then
    # Although the nnet will be trained by high resolution data, we still have to prepare normal-resolution MFCC
    # for purposes of getting alignments and/or lattices on the speed-perturbed data.
    # _sp stands for speed-perturbed

    echo "$0: preparing directory for speed-perturbed data"
    utils/data/perturb_data_dir_speed_3way.sh data/${input_data_dir} data/${input_data_dir}_sp
    mfccdir=mfcc_perturbed
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 \
      data/${input_data_dir}_sp exp/make_mfcc/${input_data_dir}_sp $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${input_data_dir}_sp exp/make_mfcc/${input_data_dir}_sp $mfccdir || exit 1;
    utils/fix_data_dir.sh data/${input_data_dir}_sp
  fi


  if [ $stage -le 2 ]; then
    #obtain the alignment of the perturbed data
    steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
      data/${input_data_dir}_sp data/lang_nosp exp/tri4 exp/tri4_ali_nodup_sp || exit 1
  fi

  clean_data_dir=${input_data_dir}_sp
else
  clean_data_dir=${input_data_dir}
fi

#wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
#      unzip rirs_noises.zip
#    fi

#    rvb_opts=()
#    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
#    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
#    rvb_opts+=(--noise-set-parameters RIRS_NOISES/pointsource_noises/noise_list)

#    python steps/data/reverberate_data_dir.py \
#      "${rvb_opts[@]}" \
#      --prefix "rev" \
#      --foreground-snrs "20:10:15:5:0" \
#      --background-snrs "20:10:15:5:0" \
#      --speech-rvb-probability 1 \
#      --pointsource-noise-addition-probability 1 \
#      --isotropic-noise-addition-probability 1 \
#      --num-replications ${num_data_reps} \
#      --max-noises-per-minute 1 \
#      --source-sampling-rate 16000 \
#      ${datadir} ${datadir}_rvb${num_data_reps}
#
#    rvb_opts+=(--rir-set-parameters "1.0, RIRS_NOISES/real_rirs_isotropic_noises/rir_list")
#    rvb_opts+=(--noise-set-parameters RIRS_NOISES/real_rirs_isotropic_noises/noise_list)
#


if [ $stage -le 3 ]; then
  if [ ! -d "simulated_rirs_8k" ]; then
    # Download the simulated RIR package with 8k sampling rate
    wget --no-check-certificate http://www.openslr.org/resources/26/sim_rir_8k.zip
    unzip sim_rir_8k.zip
  fi

  # corrupt the data to generate reverberated data 
  # this script modifies wav.scp to include the reverberation commands, the real computation will be done at the feature extraction
  # The script will automatically normalize the probability mass of the rir sets, so user just need to input the ratio of the sets
  # if --include-original-data is true, the original data will be mixed with its reverberated copies
  python steps/data/reverberate_data_dir.py \
    --prefix "rev" \
    --rir-set-parameters "0.2, simulated_rirs_8k/smallroom/rir_list" \
    --rir-set-parameters "0.2, simulated_rirs_8k/mediumroom/rir_list" \
    --rir-set-parameters "0.1, simulated_rirs_8k/largeroom/rir_list" \
    --rir-set-parameters "0.5, RIRS_NOISES/real_rirs_isotropic_noises/rir_list" \
    --foreground-snrs "20:10:15:5:0" \
    --background-snrs "20:10:15:5:0" \
    --noise-set-parameters "0.5, RIRS_NOISES/pointsource_noises/noise_list" \
    --noise-set-parameters "0.5, RIRS_NOISES/real_rirs_isotropic_noises/noise_list" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 1 \
    --isotropic-noise-addition-probability 1 \
    --num-replications $num_data_reps \
    --max-noises-per-minute 1 \
    --source-sampling-rate 8000 \
    --include-original-data true \
    data/${clean_data_dir} data/${clean_data_dir}_rvb${num_data_reps}
fi


if [ $stage -le 4 ]; then
  mfccdir=mfcc_rvb
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/swbd-$date/s5b/$mfccdir/storage $mfccdir/storage
   elif [[ $(hostname -f) == Goliath ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /media/ssd/speechtmp/multi_en-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage \
                              /media/ssd2/speechtmp/multi_en-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage \
                              /media/drive3/speechtmp/multi_en-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage \
                              $mfccdir/storage

  fi

  for dataset in ${clean_data_dir}_rvb${num_data_reps}; do
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires

    # do volume-perturbation on the training data prior to extracting hires
    # features; this helps make trained nnets more invariant to test data volume.
    utils/data/perturb_data_dir_volume.sh data/${dataset}_hires

    steps/make_mfcc.sh --nj 35 --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/${dataset} $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_hires;
  done

  #for dataset in eval2000; do
  #  # Create MFCCs for the eval set
  #  utils/copy_data_dir.sh data/$dataset/test data/${dataset}_hires/test
  #  steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 --mfcc-config conf/mfcc_hires.conf \
  #      data/${dataset}_hires/test exp/make_hires/$dataset;
  #  steps/compute_cmvn_stats.sh data/${dataset}_hires/test exp/make_hires/$dataset;
  #  utils/fix_data_dir.sh data/${dataset}_hires/test
  #done
fi

#exit 0;

# ivector extractor training
if $estimate_ivec && [ $stage -le 5 ]; then
  # Here it is good enough to train the lda_mllt transform with the clean data
  # as it only affects the diagonal GMM which is just used to initialize the full GMM
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --splice-opts "--left-context=3 --right-context=3" \
    5500 90000 data/$multi/tdnn_100k_hires \
    data/lang_nosp exp/$multi/tri4_ali $ivector_dir/tri4b
fi

train_set=${clean_data_dir}_rvb${num_data_reps}

if $estimate_ivec && [ $stage -le 6 ]; then
  # To train a diagonal UBM we don't need very much data, so use the smallest subset.
  utils/subset_data_dir.sh data/${train_set}_hires 60000 data/${train_set}_60k_hires
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 400000 \
    data/${train_set}_60k_hires 512 $ivector_dir/tri4b $ivector_dir/diag_ubm
fi

if $estimate_ivec && [ $stage -le 7 ]; then
  # iVector extractors can be sensitive to the amount of data, but this one has a
  # fairly small dim (defaults to 100) so we don't use all of it, we use just the
  # 100k subset (just under half the data).
  utils/subset_data_dir.sh data/${train_set}_hires 200000 data/${train_set}_200k_hires
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 2 \
    data/${train_set}_200k_hires $ivector_dir/diag_ubm $ivector_dir/extractor || exit 1;
fi

if [ $stage -le 8 ]; then
  # We extract iVectors on all the train_nodup data, which will be what we
  # train the system on.
  # handle per-utterance decoding well (iVector starts at zero).

  max=1
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max $max data/${train_set}_hires data/${train_set}_max${max}_hires

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/${train_set}_max${max}_hires $ivector_dir/extractor $ivector_dir/ivectors_${train_set} || exit 1;
  
  for data_set in eval2000; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
      data/${data_set}_hires/test $ivector_dir/extractor $ivector_dir/ivectors_$data_set || exit 1;
  done
fi

exit 0;

