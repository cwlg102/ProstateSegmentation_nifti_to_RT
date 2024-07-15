import os 
import pydicom 
import numpy as np

im_path = r"G:\! project\2024- ProstateSegmentation\20240125\RTSTtest\7646480\DCMData"
im_dirs_list = os.listdir(im_path)
im_dcm_list = []
for i in range(len(im_dirs_list)):
    ds = pydicom.dcmread(os.path.join(im_path, im_dirs_list[i]), force=True)
    IPPZ = float(ds.ImagePositionPatient[2])
    im_dcm_list.append((ds, IPPZ))
im_dcm_list.sort(key=lambda x:x[1])

origin = np.array([float(im_dcm_list[0][0].ImagePositionPatient[0]), float(im_dcm_list[0][0].ImagePositionPatient[1]), float(im_dcm_list[0][0].ImagePositionPatient[2])])
spacing = np.array([float(im_dcm_list[0][0].PixelSpacing[0]), float(im_dcm_list[0][0].PixelSpacing[1]), float((im_dcm_list[-1][1] - im_dcm_list[0][1])/(len(im_dcm_list) - 1))])

rtst_path = r"G:\! project\2024- ProstateSegmentation\20240125\MR_rtst\fefdafbcbd0ce22540dd1074a1e11f5eba99bd3aa1d0c9d84f43d34f93b03644\0672ca7d4807000170e8d6100395e26e16a0d62720c0c80e447ea9c9bd6d94b6.dcm"

rtst = pydicom.dcmread(rtst_path, force=True)
ref_uid = rtst.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
# ReferencedFrameOfReferenceSequence - 원래 이것도 제작해야하나.. Oncostudio에서 맞춰 줬음.

#없으면 이런식으로 제작
# if 'StructureSetROISequence' not in rtst:
#     rtst.StructureSetROISequence = []

# StructureSetROISequence 제작
name_list = ["Bladder", "Femur_Head_L", "Femur_Head_R", "Anorectum"]
color_list = [[255, 0, 255], [0, 255, 255], [250, 128, 114], [150, 23, 36]]
contour_index_list = [1, 2, 3, 5]
#contour_point_npy는... Voxel coordinate, Head First Supine & LPS coordinate (X: R-L order, Y: A-P order, Z: I-S order)
#이미지도 맞춰 줘야 ! 
#contour_hierarchy돌려서 voxel coordinate 가져오기.
contour_point_npy = np.load(r"G:\! project\2024- ProstateSegmentation\20240125\RTST_contourpoint_npy\7646480_P069.npy", allow_pickle=True)

print(contour_point_npy)
for i in range(len(name_list)):
    temp_obj = pydicom.dataset.Dataset()
    rtst.StructureSetROISequence.append(temp_obj)
    rtst.StructureSetROISequence[-1].ROINumber = (i+1)
    rtst.StructureSetROISequence[-1].ReferencedFrameOfReferenceUID = ref_uid
    rtst.StructureSetROISequence[-1].ROIName = name_list[i]
    rtst.StructureSetROISequence[-1].ROIDescription = ""
    rtst.StructureSetROISequence[-1].ROIGenerationAlgorithm = "MANUAL"

# ROIContourSequence 제작 
for i in range(len(name_list)):
    temp_obj = pydicom.dataset.Dataset()
    rtst.ROIContourSequence.append(temp_obj)
    rtst.ROIContourSequence[-1].ROIDisplayColor = color_list[i]
    rtst.ROIContourSequence[-1].ContourSequence = []
    rtst.ROIContourSequence[-1].ReferencedROINumber = (i+1)
    contour_point_volume = contour_point_npy.item().get(contour_index_list[i])
    for j in range(len(contour_point_volume)):
        contour_point_slice = contour_point_volume[j]
        
        voxel_Z_co = contour_point_slice[0][2]
        
        temp_obj_j = pydicom.dataset.Dataset()
        rtst.ROIContourSequence[-1].ContourSequence.append(temp_obj_j)
        temp_obj_j_2 = pydicom.dataset.Dataset()
        rtst.ROIContourSequence[-1].ContourSequence[-1].ContourImageSequence = [temp_obj_j_2]
        rtst.ROIContourSequence[-1].ContourSequence[-1].ContourImageSequence[0].ReferencedSOPClassUID = im_dcm_list[voxel_Z_co][0].SOPClassUID
        rtst.ROIContourSequence[-1].ContourSequence[-1].ContourImageSequence[0].ReferencedSOPInstanceUID = im_dcm_list[voxel_Z_co][0].SOPInstanceUID
        rtst.ROIContourSequence[-1].ContourSequence[-1].ContourGeometricType = "CLOSED_PLANAR"
        rtst.ROIContourSequence[-1].ContourSequence[-1].NumberOfContourPoints = 3 * len(contour_point_slice)
        contour_point_slice_float = np.copy(contour_point_slice) * spacing + origin
        contour_co = list(np.ravel(contour_point_slice_float))
        rtst.ROIContourSequence[-1].ContourSequence[-1].ContourData = contour_co

# RTROIObservationsSequence 제작 
for i in range(len(name_list)):
    temp_obj = pydicom.dataset.Dataset()
    rtst.RTROIObservationsSequence.append(temp_obj)
    rtst.RTROIObservationsSequence[-1].ObservationNumber = (i+1)
    rtst.RTROIObservationsSequence[-1].ReferencedROINumber = (i+1)
    rtst.RTROIObservationsSequence[-1].ROIObservationDescription = 'Type:Soft,Range:*/*,Fill:0,Opacity:0.0,Thickness:1,LineThickness:2,read-only:false'
    rtst.RTROIObservationsSequence[-1].RTROIInterpretedType = ""
    rtst.RTROIObservationsSequence[-1].ROIInterpreter = [""]

    
    
    
rtst.save_as(os.path.join(r"G:\! project\2024- ProstateSegmentation\20240125\RTSTtest", "test_1.dcm"))
