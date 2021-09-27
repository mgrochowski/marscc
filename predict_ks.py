from keras_segmentation_mod import predict

# wrap funkction from predict.py
def model_from_checkpoint_path(checkpoints_path, epoch=None):
    import os
    import json
    from keras_segmentation.models.all_models import model_from_name

    assert (os.path.isfile(checkpoints_path+"_config.json")
            ), "Checkpoint not found."
    model_config = json.loads(
        open(checkpoints_path+"_config.json", "r").read())
    if epoch is None:
        latest_weights = predict.find_latest_checkpoint(checkpoints_path)
    else:
        latest_weights = checkpoints_path + ".%05d" % epoch
    # assert (latest_weights is not None), "Checkpoint not found."
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'], channels=model_config['channels'])
    print("loaded weights ", latest_weights)
    status = model.load_weights(latest_weights)

    if status is not None:
        status.expect_partial()

    return model

if __name__ == '__main__':

    model = model_from_checkpoint_path('logs\\vgg_unet_2021-09-25_001534.581033\\checkpoints\\vgg_unet', 8)
    #
    pr = predict.predict(model=model,
                    inp='data/mars_data_20210923/test_0.1/images/unnamed_testing_2_patch_011_00720_00480_r0.10.png ',
                    out_fname='out.png',
                    read_image_type=0,
                    # class_names = [ "background",    "cone", "crater" ],
                    # overlay_img=True, show_legends=True
                    )

    print(pr)

    #
    # out = model.predict_segmentation(
    #     inp="../dane/coprates_1M_resize_0.2_480x480_overlap_120/images/Coprates_patch_016_01440_00720.png",
    #     out_fname="/tmp/out.png",
    #     read_image_type=0,
    #     channels=1
    # )

    import matplotlib.pyplot as plt
    plt.imshow(pr)
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(out)
    #
    #
    # o = model.predict_segmentation(
    #     inp="../dane/coprates_1M_resize_0.2_480x480_overlap_120/images/Coprates_patch_016_01440_00720.png",
    #     out_fname="/tmp/out.png" , overlay_img=True, show_legends=True,
    #     class_names = [ "Sky",    "Building", "Pole" ],
    #     read_image_type=0
    #
    # )
    #
    # # evaluating the model
    print("Validation")
    print(predict.evaluate( model=model, inp_images_dir="data/mars_data_20210923/val/images",
                                      annotations_dir="data/mars_data_20210923/val/annotations",
                                     read_image_type=0) )
    #
