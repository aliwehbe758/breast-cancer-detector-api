<ol>
    <li>
        Download postgres and pgadmin from: https://sbp.enterprisedb.com/getfile.jsp?fileid=1259178
    </li>
    <li>
        Open pgadmin:
        <ul>
            <li>
                Make the password 'sa'
            </li>
            <li>
                Create new database 'breast-cancer-detector'
            </li>
            <li>
                Right click on the database, click on restore, select 'database.backup' file from the project, then click on 'Restore'
            </li>
        </ul>
    </li>
    <li>
        Run the project in pycharm
    </li>
    <li>
when you add a model from the application:
    <ul>
        <li>
            Ignore Params py File as didn't have any implementation in the code
        </li>
        <li>
            Model py File Name should have same 'return' structure as the below:
        </li>
        <pre>
            import os
            import torch
            import timm
            from torchcam.methods import GradCAM
            def get_model(pth_file_name):
                model = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True, num_classes=3)
                # Get the absolute path of the model weights file
                current_directory = os.path.dirname(__file__)
                pth_file_path = os.path.join(current_directory, pth_file_name)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                state_dict = torch.load(pth_file_path, map_location=device)
                model.load_state_dict(state_dict)
                target_layer = model.stages[-1].blocks[-2].conv_dw
                # Initialize Grad-CAM with the target layer name
                cam_extractor = GradCAM(model, target_layer=target_layer)
        </pre>
    </ul>
    </li>
</ol>