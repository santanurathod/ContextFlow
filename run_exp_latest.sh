
# post refactoring and relative entropic addition; running 5 times
# python main.py --train_config vi_pca_C_g_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_REOT_g+p --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_REOT_CC1_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_REOT_CC2_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized

# python main.py --train_config vi_pca_C_g_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_REOT_g+p --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_REOT_CC1_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_REOT_CC2_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized

# python main.py --train_config vi_pca_C_g_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_REOT_g+p --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_REOT_CC1_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_REOT_CC2_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized

# python main.py --train_config vi_pca_C_g_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_REOT_g+p --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_REOT_CC1_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_REOT_CC2_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized

# python main.py --train_config vi_pca_C_g_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_REOT_g+p --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_REOT_CC1_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized
# python main.py --train_config vi_pca_C_g+p_REOT_CC2_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized



# everything was normalized above, but I think there's several choices for the cost function; 
# things aren't clear because spatial cost in at par with gene cost; something we don't want


python main.py --train_config vi_pca_C_g_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_g+p --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC1_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC2_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7

python main.py --train_config vi_pca_C_g_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_g+p --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC1_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC2_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7

python main.py --train_config vi_pca_C_g_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_g+p --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC1_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC2_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7

python main.py --train_config vi_pca_C_g_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_g+p --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC1_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC2_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7

python main.py --train_config vi_pca_C_g_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_g+p --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC1_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC2_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7

python main.py --train_config vi_pca_C_g_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_g+p --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC1_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC2_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7

python main.py --train_config vi_pca_C_g_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_g+p --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC1_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC2_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7

python main.py --train_config vi_pca_C_g_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_g+p --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC1_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC2_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7

python main.py --train_config vi_pca_C_g_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_g+p --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC1_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC2_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7

python main.py --train_config vi_pca_C_g_EOT --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_g+p --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC1_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7
python main.py --train_config vi_pca_C_g_REOT_CC2_g+p+c --h5ad_path GSE232025_stereoseq.h5ad --new_experiment v_everything_normalized_no_spatial_cost_otregl_0.7


