# python main_scratch.py --train_config v1_scratch_pca --h5ad_path GSE232025_stereoseq.h5ad
# python main_scratch.py --train_config v2_scratch_pca --h5ad_path GSE232025_stereoseq.h5ad
# python main_scratch.py --train_config v3_scratch_pca --h5ad_path GSE232025_stereoseq.h5ad
# python main_scratch.py --train_config v4_scratch_pca --h5ad_path GSE232025_stereoseq.h5ad
# python main_scratch.py --train_config v5_scratch_pca --h5ad_path GSE232025_stereoseq.h5ad
# python main_scratch.py --train_config v6_scratch_pca --h5ad_path GSE232025_stereoseq.h5ad
# python main_scratch.py --train_config v7_scratch_pca --h5ad_path GSE232025_stereoseq.h5ad

# python main.py --train_config v2_pca_REOT_scratch --h5ad_path GSE232025_stereoseq.h5ad

# python main.py --train_config v2_pca_REOT_CC1_scratch --h5ad_path GSE232025_stereoseq.h5ad
# python main.py --train_config v2_pca_REOT_CC2_scratch --h5ad_path GSE232025_stereoseq.h5ad

# python main.py --train_config vi_pca_C_g_REOT1_mc+lr_debug  --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_scratch_interp --interpolation True --train_idx 0 1 2 3 --test_idx 4


python main.py --train_config vi_pca_C_g_EOT --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_debug_interp --interpolation True --train_idx 0 1 2 4 --test_idx 3

# python main.py --train_config vi_pca_C_g+mc_EOT1 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+mc_EOT2 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+mc_EOT3 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+mc_EOT4 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_debug_interp --interpolation True --train_idx 0 1 2 4 --test_idx 3
# python main.py --train_config vi_pca_C_g+mc_EOT5 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+mc_EOT6 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc --interpolation True --train_idx 0 1 3 4 --test_idx 2

# python main.py --train_config vi_pca_C_g+lr_EOT1 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+lr_EOT2 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+lr_EOT3 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+lr_EOT4 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_debug_interp --interpolation True --train_idx 0 1 2 4 --test_idx 3
# python main.py --train_config vi_pca_C_g+lr_EOT5 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+lr_EOT6 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2

# python main.py --train_config vi_pca_C_g+mc+lr_EOT1 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+mc+lr_EOT2 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+mc+lr_EOT3 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+mc+lr_EOT4 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_debug_interp --interpolation True --train_idx 0 1 2 4 --test_idx 3
# python main.py --train_config vi_pca_C_g+mc+lr_EOT5 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+mc+lr_EOT6 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2

# python main.py --train_config vi_pca_C_g_REOT1_lr --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_debug_interp --interpolation True --train_idx 0 1 2 4 --test_idx 3
# python main.py --train_config vi_pca_C_g_REOT1_mc --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_debug_interp --interpolation True --train_idx 0 1 2 4 --test_idx 3
# python main.py --train_config vi_pca_C_g_REOT1_mc+lr --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_REOT_w_C_g_reg_mc+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g_REOT2_mc+lr --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_REOT_w_C_g_reg_mc+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g_REOT3_mc+lr --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_REOT_w_C_g_reg_mc+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g_REOT4_mc+lr --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_debug_interp --interpolation True --train_idx 0 1 2 4 --test_idx 3





# python main.py --train_config vi_pca_C_g_EOT --h5ad_path GSE062025_mosta_g_10000_nzp_0.1.h5ad --new_experiment v_debug_interp --interpolation True --train_idx 0 1 2 3 4 6 7 --test_idx 5

# python main.py --train_config vi_pca_C_g+mc_EOT1 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+mc_EOT2 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+mc_EOT3 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+mc_EOT4 --h5ad_path GSE062025_mosta_g_10000_nzp_0.1.h5ad --new_experiment v_debug_interp --interpolation True --train_idx 0 1 2 3 4 6 7 --test_idx 5
# python main.py --train_config vi_pca_C_g+mc_EOT5 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+mc_EOT6 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc --interpolation True --train_idx 0 1 3 4 --test_idx 2

# python main.py --train_config vi_pca_C_g+lr_EOT1 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+lr_EOT2 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+lr_EOT3 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+lr_EOT4 --h5ad_path GSE062025_mosta_g_10000_nzp_0.1.h5ad --new_experiment v_debug_interp --interpolation True --train_idx 0 1 2 3 4 6 7 --test_idx 5
# python main.py --train_config vi_pca_C_g+lr_EOT5 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+lr_EOT6 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2

# python main.py --train_config vi_pca_C_g+mc+lr_EOT1 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+mc+lr_EOT2 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+mc+lr_EOT3 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+mc+lr_EOT4 --h5ad_path GSE062025_mosta_g_10000_nzp_0.1.h5ad --new_experiment v_debug_interp --interpolation True --train_idx 0 1 2 3 4 6 7 --test_idx 5
# python main.py --train_config vi_pca_C_g+mc+lr_EOT5 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g+mc+lr_EOT6 --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_EOT_w_g+mc+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2

# python main.py --train_config vi_pca_C_g_REOT1_lr --h5ad_path GSE062025_mosta_g_10000_nzp_0.1.h5ad --new_experiment v_debug_interp --interpolation True --train_idx 0 1 2 3 4 6 7 --test_idx 5
# python main.py --train_config vi_pca_C_g_REOT1_mc --h5ad_path GSE062025_mosta_g_10000_nzp_0.1.h5ad --new_experiment v_debug_interp --interpolation True --train_idx 0 1 2 3 4 6 7 --test_idx 5
# python main.py --train_config vi_pca_C_g_REOT1_mc+lr --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_REOT_w_C_g_reg_mc+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g_REOT2_mc+lr --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_REOT_w_C_g_reg_mc+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g_REOT3_mc+lr --h5ad_path GSE232025_stereoseq_g_10000_nzp_0.1.h5ad --new_experiment v_interp_post_prior_correction_REOT_w_C_g_reg_mc+lr --interpolation True --train_idx 0 1 3 4 --test_idx 2
# python main.py --train_config vi_pca_C_g_REOT4_mc+lr --h5ad_path GSE062025_mosta_g_10000_nzp_0.1.h5ad --new_experiment v_debug_interp --interpolation True --train_idx 0 1 2 3 4 6 7 --test_idx 5
