import re
import vtk
import webcolors
import matplotlib
import numpy as np
import nibabel as nib
from fury import actor,utils
from scipy.ndimage import gaussian_filter
from fury.colormap import colormap_lookup_table
from vtkmodules.vtkRenderingCore import vtkProperty

import tslearn.metrics as tslearn
from tslearn.metrics import dtw_path
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import length, transform_streamlines

class load_3dbrain:

    def __init__(self,nifti) -> None:
        self.data = nifti.get_fdata()
        self.threshold = 45
        self.sigma = 0.5
        self.affine = nifti.affine
        self.glass_brain_actor = actor

    def loading(self):
        self.data[self.data<self.threshold] = 0
        smooth_data = gaussian_filter(self.data,sigma=self.sigma)
        self.glass_brain_actor = actor.contour_from_roi(self.data,affine = self.affine,color=[0,0,0],opacity=0.08)
        # self.set_property()
        return self.glass_brain_actor


class load_2dbrain:
    def __init__(self,nifti) -> None:
        self.data = nifti.get_fdata()
        self.affine = nifti.affine
        self.mean, self.std = self.data[self.data > 0].mean(), self.data[self.data > 0].std()
        self.value_range = (self.mean - 0.1* self.std, self.mean + 3 * self.std)

    def load_actor(self):
        slice_actor = actor.slicer(self.data,affine = self.affine,value_range=self.value_range,opacity=0.9)
        return slice_actor


class Mesh:
    def __init__(self,vtk,color_list=[]) -> None:
        self.vtk = vtk
        self.color_list = color_list

    def property(self):
        property = vtkProperty()
        property.SetColor(self.color_list)
        property.SetOpacity(1.0)
        property.SetRoughness(0.0)
        return property

    def load_mesh(self):
        property = self.property()
        actor_2 = vtk.vtkActor()
        actor_2 = utils.get_actor_from_polydata(self.vtk)
        actor_2.SetProperty(property)
        return actor_2

    def load_mesh_with_colors(self,mask,color_map):
        voxels = nib.affines.apply_affine(np.linalg.inv(mask.affine), self.vtk.points).astype(int)
        shape = mask.get_fdata().shape
        array_3d = np.zeros(shape, dtype=float)
        for index in voxels:
            array_3d[int(index[0]),int(index[1]),int(index[2])] = mask.get_fdata()[index[0],index[1],index[2]]
        roi_dict = np.delete(np.unique(array_3d), 0)
        unique_roi_surfaces = vtk.vtkAssembly()
        color_map = np.asarray(color_map)
        for i, roi in enumerate(roi_dict):
            roi_data = np.isin(array_3d,roi).astype(int)
            roi_surfaces = actor.contour_from_roi(roi_data,affine=mask.affine,color=color_map[i],opacity=1)
            unique_roi_surfaces.AddPart(roi_surfaces)
        return unique_roi_surfaces
        

class Colors:

    def __init__(self):
        self.rgb_decimal_tuple = None
    
    def get_rgb_from_color_name(self,color_name):
        try:
            rgb_tuple = webcolors.name_to_rgb(color_name)
            self.rgb_decimal_tuple = tuple(component / 255.0 for component in rgb_tuple)
            return self.rgb_decimal_tuple
        except ValueError:
            print("Invalid color name:",color_name)
            return None
    
    def hex_to_rgb(self,hex_value):
        rgb_tuple = webcolors.hex_to_rgb(hex_value) #Check Matplotlib
        self.rgb_decimal_tuple = tuple(component / 255.0 for component in rgb_tuple)
        return self.rgb_decimal_tuple

    def string_to_list(self,input_string):
        list_of_lists = []
        cleaned_string = input_string.replace('[', '').replace(']', '').replace('(','').replace(')','')  
        elements = cleaned_string.split(',')
        for element in elements:
            if element.startswith('#'):
                list_of_lists.append(self.hex_to_rgb(element))
            else:
                list_of_lists.append(self.get_rgb_from_color_name(element))
        return list_of_lists
    
    def get_tab20_color(index,type_):
        if type_=='vol':
            tab20_colors = matplotlib.cm.get_cmap('tab20')
        else:
            tab20_colors = matplotlib.cm.get_cmap('Pastel1_r')

        return matplotlib.colors.to_rgb(tab20_colors(index))

    def load_colors(self,colors_path=None):
        dic_colors = {}
        if colors_path==None: return
        with open(colors_path) as colors_file:
            lines = [line.rstrip() for line in colors_file if not line.rstrip().startswith('#')]
        for i in lines:
            match = re.search(r'[a-zA-Z]', i)
            first_alphabet_index = match.start() if match else None
            # Find the index of the last alphabetic character
            last_alphabet_index = None
            for match in re.finditer(r'[a-zA-Z]', i):
                last_alphabet_index = match.start()
            
            key = str(i[first_alphabet_index:last_alphabet_index+1])
            colors_str = i[last_alphabet_index+2:]
            colors_list = colors_str.split()
            colors_list = list(map(int, colors_list))
            rgb_tuple = colors_list[0:-1]
            rgb_decimal_tuple = tuple(component / 255.0 for component in rgb_tuple)
            dic_colors[key] = rgb_decimal_tuple

        return dic_colors

def perform_dtw(model_bundle, subject_bundle, num_segments, mask_img=None, transform=None):
    """
    This function performs Dynamic Time Warping (DTW) on two tractogram (.trk)
    files in same space.

    Args:
        tbundle (str): path to a template .trk file
        sbundle (str): Path to a subject .trk file
        num_segments (int): number of points (N+1) of template centroid to segment the bundle (N)

    Returns:
        dict: dictionary containing the corresponding points.
    """

    # reference_image = nib.load(mask_img)

    ## Transform the Template bundle to the subject space world cordinates and then to the subject voxel space cordinates:
    model_streamlines = model_bundle

    if transform is not None:
        transform_matrix = load_matrix_in_any_format(transform)
        transformed_model_bundles = transform_streamlines(model_streamlines, transform_matrix)
       
    # else:
    #     transformed_model_bundles = transform_streamlines(model_streamlines, np.linalg.inv(reference_image.affine))
    transformed_model_bundles = model_streamlines
    m_feature = ResampleFeature(nb_points=num_segments)
    m_metric = AveragePointwiseEuclideanMetric(m_feature)
    m_qb = QuickBundles(threshold=np.inf, metric=m_metric)
    m_centroid = m_qb.cluster(transformed_model_bundles).centroids
    # print('Model: Centroid length... ', np.mean([length(streamline) for streamline in m_centroid]))

    ## Transform the Subject bundle to the subject voxel cordinates:
    subject_streamlines = subject_bundle
    transformed_subject_bundles = subject_streamlines
    s_feature = ResampleFeature(nb_points=500)
    s_metric = AveragePointwiseEuclideanMetric(s_feature)
    s_qb = QuickBundles(threshold=np.inf, metric=s_metric)
    s_centroid = s_qb.cluster(transformed_subject_bundles).centroids
    # print('Subject: Centroid length... ', np.mean([length(streamline) for streamline in s_centroid]))

    ## Create multiple centroids from subject bundle using QuickBundles
    num_clusters = 100
    feature = ResampleFeature(nb_points=500)
    metric = AveragePointwiseEuclideanMetric(feature)
    qb = QuickBundles(threshold=2., metric=metric, max_nb_clusters=num_clusters)
    centroids = qb.cluster(transformed_subject_bundles).centroids

    ## Check if the centroids are flipped compared to the model centroid
    # s_centroid = reorient_streamlines(m_centroid, s_centroid)
    # centroids = reorient_streamlines(m_centroid, centroids)

    ## Compute the correspondence between the model and the subject centroids using DTW
    dtw_corres = []
    for idx, (m_centroid, s_centroid) in enumerate(zip(m_centroid, s_centroid)):
        pathDTW, similarityScore = dtw_path(m_centroid, s_centroid)
        x1, y1, z1 = m_centroid[:, 0], m_centroid[:, 1], m_centroid[:, 2]
        x2, y2, z2 = s_centroid[:, 0], s_centroid[:, 1], s_centroid[:, 2]
        corres = dict()
        for (i, j) in pathDTW:
            key = (x1[i], y1[i], z1[i])
            value = (x2[j], y2[j], z2[j])
            if key in corres:
                corres[key].append(value)
            else:
                corres[key] = [value]
        centroid_corres = []
        for key in corres.keys():
            t = len(corres[key]) // 2
            centroid_corres.append(corres[key][t])
        dtw_corres.append(np.array(centroid_corres))

    ## Establish correspondence between dtw_corres and centroids of the subject bundle
    s_corres = []
    for idx, centroid in enumerate(centroids):

        s_centroid = np.squeeze(centroid)
        s_ref  = np.squeeze(dtw_corres)
        pathDTW, similarityScore = dtw_path(s_ref, s_centroid)
        x1, y1, z1 = s_ref[:, 0], s_ref[:, 1], s_ref[:, 2]
        x2, y2, z2 = s_centroid[:, 0], s_centroid[:, 1], s_centroid[:, 2]
        corres = dict()
        for (i, j) in pathDTW:
            key = (x1[i], y1[i], z1[i])
            value = (x2[j], y2[j], z2[j])
            if key in corres:
                corres[key].append(value)
            else:
                corres[key] = [value]

        centroid_corres = []
        for key in corres.keys():
            t = len(corres[key]) // 2
            centroid_corres.append(corres[key][t])
        s_corres.append(np.array(centroid_corres))

    ## combine correspondences
    combined_corres = dtw_corres + s_corres

    ## Remove centroids that are shorter than the threshold
    data = []
    for streamline in combined_corres:
        data.append(length(streamline))  
    mean_length = np.mean(data)
    std_length = np.std(data)
    # print("Average streamlines length", np.mean(data))
    # print("Standard deviation", std_length)
    threshold = mean_length - 1 * std_length
    indices = np.where(data < threshold)
    final_corres = [sl for idx, sl in enumerate(combined_corres) if idx not in indices[0]]

    ## Compute pairwise distances between corresponding points of the final centroids
    corresponding_points = np.array(final_corres)
    pairwise_distances = np.zeros((corresponding_points.shape[1], corresponding_points.shape[0], corresponding_points.shape[0]))
    for i in range(corresponding_points.shape[1]):
        for j in range(corresponding_points.shape[0]):
            for k in range(j + 1, corresponding_points.shape[0]):
                pairwise_distances[i, j, k] = np.linalg.norm(corresponding_points[j, i] - corresponding_points[k, i])
    pairwise_distances[pairwise_distances == 0] = np.nan
    mean_distances = np.nanmean(pairwise_distances, axis=(1, 2))
    std_distances = np.nanstd(pairwise_distances, axis=(1, 2))
    excluded_idx = np.where(std_distances <= 3.5)[0]

    ## Filter the final_corres based on pairwise distances that have std <= 3.5
    excluded_start = excluded_idx[0]
    excluded_end = excluded_idx[-1]

    filtered_arrays = []
    for idx, array in enumerate(final_corres):
        combined_array = []
        if excluded_start > 1:
            start_point = array[0]
            end_point = array[excluded_start]
            side_1_points = np.linspace(start_point, end_point, excluded_start + 1)[1:-1]
            combined_array.extend(array[0:1])
            combined_array.extend(side_1_points)
        elif excluded_start <= 1:
            combined_array.extend(array[0:excluded_start])
        combined_array.extend(array[excluded_start:excluded_end+1])
        if num_segments - excluded_end > 1:
            start_point = array[excluded_end]
            end_point = array[-1]
            side_2_points = np.linspace(start_point, end_point, num_segments - excluded_end)[1:-1]
            combined_array.extend(side_2_points)
            combined_array.extend(array[-1:])
        elif num_segments - excluded_end == 1:
            combined_array.extend(array[-1:])

        filtered_arrays.append(np.array(combined_array))
    # print("Total number filtered centroids:", len(filtered_arrays))
    return filtered_arrays