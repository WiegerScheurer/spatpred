

# Define the colors for the line plots
colors = ['r', 'g', 'b', 'c']
rois = ['V1', 'V2', 'V3', 'V4']  

# Create 1 row and 4 columns for the plots
fig, axes = plt.subplots(1, 4, figsize=(15, 10))

for layer in range(1,5):
    for n_roi, roi in enumerate(rois):   
        print(f'')
        
        this_y, this_xyzs = NSP.analyse.load_y('subj01', roi, y_dict, xyz_to_vox, roi_masks, prf_dict, n_voxels = 'all', start_img = 0, n_imgs = 30000)
        this_X = np.load(f'/home/rfpred/data/custom_files/subj01/pred/featmaps/Aunet_gt_feats_{layer}.npy')
        n_imgs = 100
        # X = Xcnn_stack
        X = NSP.utils.get_zscore(Xcnn4, print_ars='n')
        alf = .01
        cv = 15
        results = NSP.analyse.evaluate_model(this_X[:n_imgs,:], this_y[:n_imgs], alpha=alf, cv=cv, extra_stats=False)

        n_voxels = this_y.shape[1]
        cv_r2_all = np.zeros((n_voxels, 4))
        for vox in range(n_voxels):
            for idx in range(3):
                cv_r2_all[vox,idx] = this_xyzs[vox][0][idx]
            cv_r2_all[vox,3] = results['cross_validation_scores'][vox]

        cv_sorted = NSP.utils.sort_by_column(cv_r2_all, 3, top_n=n_voxels)
        # Plot the line plot in the corresponding subplot with the corresponding color
        line = axes[layer-1].plot(cv_sorted[:,3]*100, color=colors[n_roi], label=roi)
        axes[layer-1].set_title(f'Layer {layer}')
        axes[layer-1].set_xlabel('Number of voxels')
        axes[layer-1].set_ylabel('Cross-validated R2 in %')
        axes[layer-1].set_ylim([-10, 25])

    # Add a legend to the first subplot
    if layer == 1:
        axes[layer-1].legend()
    
    # Remove the y-axis for the right 3 plots
    if layer > 1:
        axes[layer-1].set_yticks([])
        axes[layer-1].set_ylabel('')  # Remove the y-axis label

# Save the figure as a .png file
plt.savefig('/home/rfpred/imgs/UNet_featmaps_reg.png')

# Display the plots
plt.show()