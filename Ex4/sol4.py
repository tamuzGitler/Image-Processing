# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged 
################################ Imports ####################################

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, convolve, map_coordinates
import shutil
from imageio import imwrite
import sol4_utils
from numpy.linalg import inv

################################ Constants ####################################

GRAY = "gray"
POWER_FACTOR = 2
K = 0.04
KERNEL_SIZE = 3
DESC_RAD = 3
DESCRIPTOR_SIZE = 7
RADIUS = 13
LAST_COLL = -1
SECOND_MAX_INDEX = 2
OUTLIERS_LW = .1
INLINERS_LW = .8


################################ Implement Functions ##########################
######## Task 3.1 ########

def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    Ix, Iy = get_derivatives(im)
    Ix2 = np.power(Ix, POWER_FACTOR)  # Ix^2
    Iy2 = np.power(Iy, POWER_FACTOR)  # Iy^2
    Ixy = Ix * Iy
    Ix2 = sol4_utils.blur_spatial(Ix2, kernel_size=KERNEL_SIZE)
    Iy2 = sol4_utils.blur_spatial(Iy2, kernel_size=KERNEL_SIZE)
    Ixy = sol4_utils.blur_spatial(Ixy, kernel_size=KERNEL_SIZE)

    # M presentation:  [[Ix2, IxIy], [IxIy, Iy2]])
    detM = (Ix2 * Iy2) - np.power(Ixy, POWER_FACTOR)
    traceM = Ix2 + Iy2  # trace - sum of diagonals of the array
    R = detM - K * np.power(traceM, POWER_FACTOR)

    is_local_maximum_point = non_maximum_suppression(R)  # gets binary image containing the local maximum points
    corners_coordinates = np.argwhere(is_local_maximum_point)  # gets (x,y) coordinates where value is True=corner
    corners_coordinates = np.array([corners_coordinates[:, 1], corners_coordinates[:,
                                                               0]]).T  # Note the coordinate order is (x,y), as opposed to the order when indexing an array which is (row,column)
    return corners_coordinates


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    # note1: im is 3rd level pyramid image
    # note2: desc_rad should be set to 3
    # note3: assume pos in corrdinate of corners in 3rd level pyramid
    K = 2 * desc_rad + 1  # descriptor window
    N = pos.shape[0]  # number of coordinates
    descriptors = np.zeros([N, K, K])
    for i in range(N):  # run over each corner coordinate
        # Working with matrix coordinate and not image coordninate.
        col, row = pos[i, 0], pos[i, 1]  # get y=rows,x=cols position in IMAGE coordinate, but in matrix y=col and x=row
        row_indices = np.arange(row - desc_rad, row + desc_rad + 1)  # create row descriptor index's
        col_indices = np.arange(col - desc_rad, col + desc_rad + 1)  # get col descriptor index's
        col_mat, row_mat = np.meshgrid(col_indices, row_indices)  # create x,y indices matrix's around (x,y) position
        descriptor = map_coordinates(im, (row_mat, col_mat), order=1, prefilter=False)  # interpolate
        norm_mean_descriptor = np.linalg.norm(descriptor - descriptor.mean())
        if (norm_mean_descriptor != 0):
            descriptor = (descriptor - descriptor.mean()) / norm_mean_descriptor
        else:
            descriptor = np.zeros((K, K))
        descriptors[i, :, :] = descriptor

    return descriptors


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    pos = spread_out_corners(pyr[0], m=DESCRIPTOR_SIZE, n=DESCRIPTOR_SIZE, radius=RADIUS)  # find corner points
    level2_pos = 0.25 * pos  # transform coordniate from original image (level 0) to im (level 2), (2 ** -2)=0.25
    descriptors = sample_descriptor(im=pyr[2], pos=level2_pos, desc_rad=DESC_RAD)
    return [pos, descriptors]  # returns list as asked to9


######## Task 3.1 Helpers ########

def get_derivatives(im):
    derivate_filter = np.array([1, 0, -1]).reshape([1, 3])
    Ix = convolve(im, derivate_filter)  # Horizontal Derivative - x
    Iy = convolve(im, derivate_filter.T)  # Vertical Derivative - y
    return Ix, Iy


######## Task 3.2 ########

def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    # Save shapes of descriptors
    M = desc1.shape[0]
    N = desc2.shape[0]
    K = desc1.shape[1]

    # Calc S by falttering the descriptors
    desc1flat = desc1.reshape(M, K * K)  # each row is flattern descriptor
    desc2flat = desc2.reshape(N, K * K)  # same
    S = desc1flat @ desc2flat.T  # create matrix where S[i,j]= desc1i * desc2j

    # get the second maximum value of each row and col
    row_second_greatest = np.partition(S, -SECOND_MAX_INDEX)[:, -SECOND_MAX_INDEX].reshape(M,
                                                                                           1)  # reshape for later use

    col_second_greatest = np.partition(S, -SECOND_MAX_INDEX, axis=0)[-SECOND_MAX_INDEX,
                          :]  # get the second greatest value of each column

    return np.where((S > min_score) & (S >= row_second_greatest) & (
            S >= col_second_greatest))  # return matching indices with 3 conditions


######## Task 3.3 ########

def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    N = pos1.shape[0]
    oneVector = np.ones(N).reshape(N, 1)
    pos1WithOne = np.append(pos1, oneVector, axis=1).T  # so it will look like in ex4 pdf [x1,y1,1] column vector
    transformingPos1 = (H12 @ pos1WithOne).T
    z = transformingPos1[:, LAST_COLL]  # get last column which contains z values
    transformed_pos1 = (transformingPos1[:, :2].T / z).T
    return transformed_pos1


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    N = points1.shape[0]
    inliers = []  # indices
    num_of_inliners = 0
    size = 2
    if translation_only:  # only one pair of points is needed
        size = 1
    for i in range(num_iter):

        random_indexs = np.random.choice(np.arange(N), size=size, replace=False)  # get random index's
        P1J = np.array(points1[random_indexs])  # get points by random_indexs from points1
        P2J = np.array(points2[random_indexs])  # get points by random_indexs from points2

        H12 = estimate_rigid_transform(P1J, P2J, translation_only=False)  # Compute the homography

        transformedPoints1 = apply_homography(points1, H12)

        E = np.power(np.linalg.norm(transformedPoints1 - points2, axis=1), 2)  # calc error
        matches_indexs = np.argwhere(E < inlier_tol)

        if matches_indexs.shape[
            0] > num_of_inliners:  # check if current inliners points are bigger than previus maximum
            inliers = matches_indexs  # save inliners for later
            num_of_inliners = matches_indexs.shape[0]  # save number of new best inliners

    inliers = inliers.squeeze()  # squeeze to make them good for index use
    inliersP1 = points1[inliers]  # get points by inliners indexs
    inliersP2 = points2[inliers]  # get points by inliners indexs
    H12 = estimate_rigid_transform(inliersP1, inliersP2, translation_only=translation_only)  # Compute the homography
    return [H12, inliers]


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    N = points1.shape[0]
    outliers = np.delete(np.arange(N), inliers)  # get outliers indices
    image = np.hstack((im1, im2))  # combine the two images
    points2[:, 0] += im1.shape[
        1]  # shifts points2 by the number of columns in im1, so it will correspond to image indexs

    plot_lines(points1, points2, indices=inliers, color="yellow", lw=INLINERS_LW)
    plot_lines(points1, points2, indices=outliers, color="blue", lw=OUTLIERS_LW)
    plt.imshow(image, cmap=GRAY)
    plt.show()


######## Task 3.3 helpers ########

def plot_lines(points1, points2, indices, color, lw):
    for i in indices:
        x = [points1[i, 0], points2[i, 0]]
        y = [points1[i, 1], points2[i, 1]]
        plt.plot(x, y, mfc='r', c=color, lw=lw, ms=3, marker='o')


######## Task 3.4 ########

def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    M = len(H_succesive)
    H2m = [np.eye(3)]
    i = m - 1

    # fill left side homographies
    while (i >= 0):
        new_homog = H2m[0] @ H_succesive[i]
        H2m.insert(0, new_homog)
        i -= 1

    # fill right side homographies
    i = m
    while (i < M):
        H2m.append(H2m[i] @ inv(H_succesive[i]))
        i += 1

    # normalize
    for h in H2m:
        h /= h[2, 2]

    return H2m


######## Task 4 ########


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    # remember coordinate of image are reversed
    image_corners = np.array(
        [[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]])  # calc corner positions (w-1,h-1 because we start at 0)

    transformed_corners = apply_homography(image_corners, homography)  # get new corners after applying homography
    # still image coordinate [x-col,y-row]
    x_coordinates, y_coordinates = transformed_corners[:, 0], transformed_corners[:, 1]  # seperaete x and y coordinate
    # calculate bounding box corners
    Xmin = np.min(x_coordinates)
    Ymin = np.min(y_coordinates)
    Xmax = np.max(x_coordinates)
    Ymax = np.max(y_coordinates)

    bounding_box = np.array([[Xmin, Ymin], [Xmax, Ymax]]).astype(int)  # return array should be type np.int
    return bounding_box


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    # doing backward warping
    inv_homography = inv(homography)  # calc inverse homogorphy for back wraping
    #### image coordinates ####
    h, w = image.shape  # get height and width of image
    top_left_corner, bottom_right_corner = compute_bounding_box(homography, w,
                                                                h)  # get bounding after applying homography
    # get corners index's
    widthMin, heightMin = top_left_corner
    widthMax, heightMax = bottom_right_corner
    warpedWidth = np.arange(widthMin, widthMax)  # create col index's
    warpedHeight = np.arange(heightMin, heightMax)  # create row index's
    col_mat, row_mat = np.meshgrid(warpedWidth, warpedHeight)

    mat_shape = col_mat.shape
    col_mat = col_mat.flat  # flatting to stack points easily
    row_mat = row_mat.flat  # flatting to stack points easily
    pos1 = np.stack((col_mat, row_mat), axis=-1)  # (col_mat, row_mat) - still image coordinates
    transformed_pos1 = apply_homography(pos1, inv_homography)

    #### matrix coordinates ####
    row_matrix = transformed_pos1[:, 1].reshape(mat_shape)
    col_matrix = transformed_pos1[:, 0].reshape(mat_shape)  # TODO maybe oposite -0 and not 1
    wraped_image = map_coordinates(image, (row_matrix, col_matrix), order=1, prefilter=False)
    return wraped_image


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images, bonus=False):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.bonus = bonus
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        if self.bonus:
            self.generate_panoramic_images_bonus(number_of_panoramas)
        else:
            self.generate_panoramic_images_normal(number_of_panoramas)

    def generate_panoramic_images_normal(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def generate_panoramic_images_bonus(self, number_of_panoramas):
        """
        The bonus
        :param number_of_panoramas: how many different slices to take from each input image
        """
        pass

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()
