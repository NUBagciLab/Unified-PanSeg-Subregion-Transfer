"""
Clean nnUNet inference implementation for single NIfTI file.
This is a simplified version that directly uses nnUNet's core inference components
without shell command calls.

This version is FIXED to use nnUNetSubregionTrainer architecture and nnUNetSubregionTrainer.
No dynamic trainer lookup - directly uses extension/nnUNetTrainer/nnUNetSubregionTrainer.py
"""

import os
import sys
import nibabel as nib
from nibabel.orientations import ornt_transform,io_orientation
from typing import Optional, Union, Tuple

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, join

# import nnunetv2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

# Add extension directory to path for local imports
_extension_dir = os.path.dirname(os.path.abspath(__file__))
if _extension_dir not in sys.path:
    sys.path.insert(0, _extension_dir)

# Directly import nnUNetSubregionTrainer (fixed to use nnUNetSubregionTrainer)
from nnUNetTrainer.nnUNetMultiModalPancreasTrainer import nnUNetMultiModalPancreasTrainer

# Import fixed export function
from export_prediction_fixed import export_prediction_from_logits
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape


class CleanInference:
    
    def __init__(
        self,
        model_folder: str,
        device: torch.device = torch.device('cpu'),
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_folder: Path to the trained model folder (contains dataset.json, plans.json, checkpoint_best.pth)
                         Expected structure:
                         - model_folder/
                           - checkpoint_best.pth
                           - dataset.json
                           - plans.json
            device: torch device ('cpu', 'cuda', 'mps')
            tile_step_size: Step size for sliding window (0.5 is recommended)
            use_gaussian: Whether to use Gaussian weighting for overlapping patches
            use_mirroring: Whether to use test-time augmentation (mirroring)
            verbose: Whether to print detailed information
        """
        self.model_folder = model_folder
        self.device = device
        self.verbose = verbose
        
        # Initialize predictor
        self.predictor = nnUNetPredictor(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_device=(device.type == 'cuda'),
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose,
            allow_tqdm=True
        )
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load model configuration and weights."""
        if self.verbose:
            print(f"Loading model from: {self.model_folder}")
        
        # Load dataset.json and plans.json
        dataset_json_path = join(self.model_folder, 'dataset.json')
        plans_json_path = join(self.model_folder, 'plans.json')
        
        if not os.path.exists(dataset_json_path):
            raise FileNotFoundError(f"dataset.json not found at {dataset_json_path}")
        if not os.path.exists(plans_json_path):
            raise FileNotFoundError(f"plans.json not found at {plans_json_path}")
        
        dataset_json = load_json(dataset_json_path)
        plans = load_json(plans_json_path)
        plans_manager = PlansManager(plans)
        
        # Load checkpoint (simplified: checkpoint is directly in model_folder)
        checkpoint_path = join(self.model_folder, 'checkpoint_best.pth')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}\n"
                f"Expected structure: model_folder/checkpoint_best.pth"
            )
        
        if self.verbose:
            print(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        
        configuration_name = checkpoint['init_args']['configuration']
        inference_allowed_mirroring_axes = checkpoint.get('inference_allowed_mirroring_axes', None)
        
        # Get configuration manager
        configuration_manager = plans_manager.get_configuration(configuration_name)
        
        # Determine number of input channels
        num_input_channels = determine_num_input_channels(
            plans_manager, configuration_manager, dataset_json
        )
        
        if self.verbose:
            print(f"Building network architecture: {configuration_manager.network_arch_class_name}")
        
        # Build network architecture using nnUNetSubregionTrainer
        network = nnUNetMultiModalPancreasTrainer.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )
        
        if self.verbose:
            print(f"Network built: {type(network).__name__}")
        
        # Load weights
        network.load_state_dict(checkpoint['network_weights'])
        network.eval()
        
        # Store in predictor
        self.predictor.plans_manager = plans_manager
        self.predictor.configuration_manager = configuration_manager
        self.predictor.list_of_parameters = [checkpoint['network_weights']]
        self.predictor.network = network
        self.predictor.dataset_json = dataset_json
        self.predictor.trainer_name = 'nnUNetMultiModalPancreasTrainer'  # Fixed trainer name
        self.predictor.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.predictor.label_manager = plans_manager.get_label_manager(dataset_json)
        
        if self.verbose:
            print(f"Model loaded successfully!")
            print(f"  Trainer: nnUNetMultiModalPancreasTrainer (fixed)")
            print(f"  Network type: {type(network).__name__}")
            print(f"  Configuration: {configuration_name}")
            print(f"  Input channels: {num_input_channels}")
            print(f"  Output heads: {plans_manager.get_label_manager(dataset_json).num_segmentation_heads}")
    
    def predict_single_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        save_probabilities: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict segmentation for a single NIfTI file.
        
        Args:
            input_file: Path to input NIfTI file (.nii.gz)
            output_file: Path to save output segmentation (if None, returns array)
            save_probabilities: Whether to save/return probability maps
            
        Returns:
            If output_file is None: segmentation array (or tuple with probabilities if save_probabilities=True)
            If output_file is provided: None (saves to file)
        """
        if self.verbose:
            print(f"\nProcessing: {input_file}")
        
        # Get image reader/writer class from plans
        image_reader_writer = self.predictor.plans_manager.image_reader_writer_class()
        
        # Read image
        if self.verbose:
            print("Reading image...")
        image_array, image_properties = image_reader_writer.read_images([input_file])
        # read_images returns (np.ndarray, dict), not (list, list)
        # image_array is already a numpy array with shape (C, H, W, D) or (C, H, W)
        # image_properties is a dictionary with metadata
        image = image_array  # Shape: (C, H, W, D) or (C, H, W)
        properties = image_properties  # Dictionary with spacing, origin, etc.
        
        # IMPORTANT: properties must contain 'nibabel_stuff' or 'sitk_stuff' for correct orientation
        # run_case_npy will preserve these keys, so they'll be available in data_properties for export
        # print(properties)
        if self.verbose:
            print(f"  Image shape: {image.shape}")
            print(f"  Spacing: {properties.get('spacing', 'N/A')}")
            # Check if orientation info is preserved
            if 'nibabel_stuff' in properties:
                print(f"  Found nibabel_stuff (orientation will be preserved)")
                print(f"    original_affine shape: {properties['nibabel_stuff']['original_affine'].shape}")
                print(f"    reoriented_affine shape: {properties['nibabel_stuff']['reoriented_affine'].shape}")
            elif 'sitk_stuff' in properties:
                print(f"  Found sitk_stuff (orientation will be preserved)")
            else:
                print(f"  WARNING: No orientation info found in properties!")
        
        # Preprocess
        if self.verbose:
            print("Preprocessing...")
        
        # Get preprocessor
        preprocessor = self.predictor.configuration_manager.preprocessor_class(
            verbose=self.verbose
        )
        
        # Run preprocessing
        # image should be (C, H, W, D) format, seg is None for inference
        # IMPORTANT: run_case_npy modifies properties in-place but preserves original keys
        # like 'nibabel_stuff' or 'sitk_stuff' which are needed for correct orientation in export
        data, seg, data_properties = preprocessor.run_case_npy(
            image,
            None,  # No previous stage segmentation
            properties,
            self.predictor.plans_manager,
            self.predictor.configuration_manager,
            self.predictor.dataset_json
        )

        # Verify that orientation info is preserved after preprocessing
        if self.verbose:
            if 'nibabel_stuff' in data_properties:
                print(f"  Orientation info preserved: nibabel_stuff found")
            elif 'sitk_stuff' in data_properties:
                print(f"  Orientation info preserved: sitk_stuff found")
            else:
                print(f"  WARNING: Orientation info missing in data_properties!")
                print(f"  Available keys: {list(data_properties.keys())}")
        
        # Convert to torch tensor
        data_tensor = torch.from_numpy(data).to(dtype=torch.float32)
        
        if self.verbose:
            print(f"  Preprocessed shape: {data_tensor.shape}")
        
        # Predict
        if self.verbose:
            print("Running inference...")
            print(f"  Input data_tensor shape: {data_tensor.shape}")
            print(f"  use_mirroring: {self.predictor.use_mirroring}")
            print(f"  allowed_mirroring_axes: {self.predictor.allowed_mirroring_axes}")
        print("data_tensor.shape", data_tensor.shape)
        with torch.no_grad():
            predicted_logits = self.predictor.predict_logits_from_preprocessed_data(data_tensor).cpu()

        if self.verbose:
            print(f"  Prediction logits shape: {predicted_logits.shape}")
        
        # Post-process and export
        if self.verbose:
            print("Post-processing and exporting...")
            # Final check before export
            io_class = self.predictor.plans_manager.image_reader_writer_class()
            # print(f"  Image IO class: {io_class.__name__}")
            if 'nibabel_stuff' in data_properties:
                print(f"  ✓ nibabel_stuff found - orientation will be restored")
                print(f"    original_affine: {data_properties['nibabel_stuff'].get('original_affine', 'N/A') is not None}")
                print(f"    reoriented_affine: {data_properties['nibabel_stuff'].get('reoriented_affine', 'N/A') is not None}")
            elif 'sitk_stuff' in data_properties:
                print(f"  ✓ sitk_stuff found - orientation will be restored")
            else:
                print(f"  ✗ WARNING: No orientation info in data_properties!")
                print(f"    Available keys: {list(data_properties.keys())}")
        

        # Save to file
        # export_prediction_from_logits expects output_file_truncated (without extension)
        # and will automatically add file_ending (e.g., .nii.gz)
        # So we need to remove the extension from output_file if it exists
        file_ending = self.predictor.dataset_json.get('file_ending', '.nii.gz')
        
        # Remove file_ending from output_file if present
        if output_file.endswith(file_ending):
            output_file_truncated = output_file[:-len(file_ending)]
        else:
            # Also check for common extensions
            for ext in ['.nii.gz', '.nii', '.nrrd', '.mha']:
                if output_file.endswith(ext):
                    output_file_truncated = output_file[:-len(ext)]
                    break
            else:
                # No extension found, use as is
                output_file_truncated = output_file
        
        if self.verbose:
            print(f"  Output file (truncated): {output_file_truncated}")
            print(f"  Will add file ending: {file_ending}")
        
        # The actual output file will be output_file_truncated + file_ending
        actual_output_file = output_file_truncated + file_ending
        affine_reoriented = data_properties['nibabel_stuff']['reoriented_affine']
        affine_original = data_properties['nibabel_stuff']['original_affine']
        original_orientation = io_orientation(affine_original)
        reoriented_orientation = io_orientation(affine_reoriented)
        from_re_or_to_original = ornt_transform(reoriented_orientation,original_orientation)

        if save_probabilities:
            if self.verbose:
                print("Saving probability map NIfTI (after softmax/sigmoid, before final segmentation).")
            label_manager = self.predictor.plans_manager.get_label_manager(self.predictor.dataset_json)
            _, probabilities_final = convert_predicted_logits_to_segmentation_with_correct_shape(
                predicted_logits,
                self.predictor.plans_manager,
                self.predictor.configuration_manager,
                label_manager,
                data_properties,
                return_probabilities=True,
            )
            # nnU-Net probabilities are (C, Z, Y, X).
            # For binary segmentation we save only foreground probability (channel 1)
            # to avoid confusion with background probability (channel 0).
            if probabilities_final.shape[0] == 2:
                probabilities_nifti = probabilities_final[1].transpose(2, 1, 0).astype(np.float32, copy=False)
            else:
                # For multi-class tasks we keep all channels as 4D NIfTI: (X, Y, Z, C)
                probabilities_nifti = probabilities_final.transpose(3, 2, 1, 0).astype(np.float32, copy=False)

            prob_nib = nib.Nifti1Image(probabilities_nifti, affine_reoriented)
            prob_nib = prob_nib.as_reoriented(from_re_or_to_original)
            # Keep pixdim/qform/sform consistent with the reoriented affine.
            prob_nib.set_qform(prob_nib.affine, code=1)
            prob_nib.set_sform(prob_nib.affine, code=1)
            nib.save(prob_nib, actual_output_file)
            return

        # export_prediction_from_logits will use the image_reader_writer_class from plans
        # which should be NibabelIOWithReorient, and it will use data_properties['nibabel_stuff']
        # to restore the original orientation
        segmentation_final = export_prediction_from_logits(
            predicted_logits,
            data_properties,  # Contains nibabel_stuff or sitk_stuff for orientation restoration
            self.predictor.configuration_manager,
            self.predictor.plans_manager,
            self.predictor.dataset_json,
            output_file_truncated,  # Without extension
        )
        
        # DEBUG: Check data format after export_prediction_from_logits
        if self.verbose:
            print(f"\n=== DEBUG: Data format check ===")
            print(f"segmentation_final shape: {segmentation_final.shape}")
            print(f"segmentation_final dtype: {segmentation_final.dtype}")
            print(f"transpose_backward: {self.predictor.plans_manager.transpose_backward}")
            print(f"transpose_forward: {self.predictor.plans_manager.transpose_forward}")
            print(f"Original image shape (from properties): {properties.get('shape_before_cropping', 'N/A')}")
            print(f"Shape after cropping (from data_properties): {data_properties.get('shape_before_cropping', 'N/A')}")
            # Check first and last slice to understand orientation
            if len(segmentation_final.shape) == 3:
                print(f"First slice (index 0) sum: {segmentation_final[0].sum()}")
                print(f"Last slice (index -1) sum: {segmentation_final[-1].sum()}")
                print(f"First slice along axis 0 shape: {segmentation_final[0].shape}")
                print(f"First slice along axis -1 shape: {segmentation_final[:,:,0].shape}")
        
        # nib format:x,y,z , nnunet output format:z,y,x
        # z,y,x --> x,y,z
        segmentation_final_before_transpose = segmentation_final.copy() if self.verbose else None
        segmentation_final = segmentation_final.transpose(2,1,0)
        
        if self.verbose:
            print(f"\nAfter transpose(2,1,0):")
            print(f"  segmentation_final shape: {segmentation_final.shape}")
            if segmentation_final_before_transpose is not None:
                print(f"  Before transpose shape: {segmentation_final_before_transpose.shape}")
                # Compare first/last slices to check if transpose is correct
                print(f"  Before transpose - first slice sum: {segmentation_final_before_transpose[0].sum()}")
                print(f"  After transpose - first slice sum: {segmentation_final[:,:,0].sum()}")
        
        if self.verbose:
            print(f"\n=== Orientation info ===")
            print(f"affine_reoriented shape: {affine_reoriented.shape}")
            print(f"affine_original shape: {affine_original.shape}")
        
        seg_final_nib = nib.Nifti1Image(segmentation_final, affine_reoriented)
        seg_final_nib = seg_final_nib.as_reoriented(from_re_or_to_original)
        # Keep pixdim/qform/sform consistent with the reoriented affine.
        seg_final_nib.set_qform(seg_final_nib.affine, code=1)
        seg_final_nib.set_sform(seg_final_nib.affine, code=1)
        nib.save(seg_final_nib, actual_output_file)

        # seg_nib = nib.Nifti1Image(segmentation_final, properties['affine'])



def run_inference(
    input_file: str,
    model_folder: str,
    output_file: str,
    device: str = 'cpu',
    verbose: bool = True
):
    """
    Convenience function to run inference on a single file.
    
    Args:
        input_file: Path to input NIfTI file
        model_folder: Path to model folder (contains checkpoint_best.pth, dataset.json, plans.json)
        output_file: Path to save output segmentation
        device: Device ('cpu', 'cuda', 'mps')
        verbose: Whether to print progress
    """
    # Convert device string to torch.device
    if device == 'cpu':
        torch_device = torch.device('cpu')
    elif device == 'cuda':
        torch_device = torch.device('cuda')
    elif device == 'mps':
        torch_device = torch.device('mps')
    else:
        raise ValueError(f"Unknown device: {device}")
    
    # Initialize inference engine
    inference = CleanInference(
        model_folder=model_folder,
        device=torch_device,
        verbose=verbose
    )
    
    # Run prediction
    inference.predict_single_file(
        input_file=input_file,
        output_file=output_file,
        save_probabilities=False
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean nnUNet inference for single NIfTI file (fixed to nnUNetSubregionTrainer)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model folder structure expected:
  model_folder/
    - checkpoint_best.pth
    - dataset.json
    - plans.json
        """
    )
    parser.add_argument("-i", "--input", required=True, help="Input NIfTI file (.nii.gz)")
    parser.add_argument("-o", "--output", required=True, help="Output segmentation file (.nii.gz)")
    parser.add_argument("-m", "--model", default="Model",
                       help="Model folder path (contains checkpoint_best.pth, dataset.json, plans.json)")
    parser.add_argument("-d", "--device", default="cuda", choices=["cpu", "cuda", "mps"], help="Device")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    run_inference(
        input_file=args.input,
        model_folder=args.model,
        output_file=args.output,
        device=args.device,
        verbose=args.verbose
    )

