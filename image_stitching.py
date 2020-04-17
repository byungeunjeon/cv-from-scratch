import numpy as np
from canny import *

"""
   INTEREST POINT OPERATOR (12 Points Implementation + 3 Points Write-up)

   Implement an interest point operator of your choice.

   Your operator could be:

   (A) The Harris corner detector (Szeliski 4.1.1)

               OR

   (B) The Difference-of-Gaussians (DoG) operator defined in:
       Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", 2004.
       https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

               OR

   (C) Any of the alternative interest point operators appearing in
       publications referenced in Szeliski or in lecture

              OR

   (D) A custom operator of your own design

   You implementation should return locations of the interest points in the
   form of (x,y) pixel coordinates, as well as a real-valued score for each
   interest point.  Greater scores indicate a stronger detector response.

   In addition, be sure to apply some form of spatial non-maximum suppression
   prior to returning interest points.

   Whichever of these options you choose, there is flexibility in the exact
   implementation, notably in regard to:

   (1) Scale

       At what scale (e.g. over what size of local patch) do you operate?

       You may optionally vary this according to an input scale argument.

       We will test your implementation at the default scale = 1.0, so you
       should make a reasonable choice for how to translate scale value 1.0
       into a size measured in pixels.

   (2) Nonmaximum suppression

       What strategy do you use for nonmaximum suppression?

       A simple (and sufficient) choice is to apply nonmaximum suppression
       over a local region.  In this case, over how large of a local region do
       you suppress?  How does that tie into the scale of your operator?

   For making these, and any other design choices, keep in mind a target of
   obtaining a few hundred interest points on the examples included with
   this assignment, with enough repeatability to have a large number of
   reliable matches between different views.

   If you detect more interest points than the requested maximum (given by
   the max_points argument), return only the max_points highest scoring ones.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      image       - a grayscale image in the form of a 2D numpy array
      max_points  - maximum number of interest points to return
      scale       - (optional, for your use only) scale factor at which to
                    detect interest points

   Returns:
      xs          - numpy array of shape (N,) containing x-coordinates of the
                    N detected interest points (N <= max_points)
      ys          - numpy array of shape (N,) containing y-coordinates
      scores      - numpy array of shape (N,) containing a real-valued
                    measurement of the relative strength of each interest point
                    (e.g. corner detector criterion OR DoG operator magnitude)
"""
def find_interest_points(image, max_points = 200, scale = 1.0):
   # check that image is grayscale
   assert image.ndim == 2, 'image should be grayscale'
   ##########################################################################
   # TODO: YOUR CODE HERE

   # optional blurring
   img = conv_2d_gaussian(image, sigma=1)

   # derivatives
   dx, dy = sobel_gradients(img)

   # second moment
   IxIx = conv_2d_gaussian(dx * dx, sigma=1)
   IxIy = conv_2d_gaussian(dx * dy, sigma=1)
   IyIy = conv_2d_gaussian(dy * dy, sigma=1)

   # cornerness function
   detM = IxIx * IyIy - IxIy ** 2
   traceM = IxIx + IyIy
   alpha = 0.05
   har = detM - alpha * (traceM ** 2)

   # nonmax suppression
   off = int(scale)
   threshold = 0

   cr, cc = np.where(har > threshold)
   cn = cr.shape[0]
   cv = np.zeros(cn, )
   for i in range(cn): 
       cv[i] = har[cr[i], cc[i]]
   oks = np.ones(har.shape)

   idx = np.argsort(cv)[::-1]
   rows = []
   cols = []
   vals = []
   cnt = 0

   for i in idx:
       if cnt >= max_points:
           break
       if oks[cr[i], cc[i]] == 1:
           rows.append(cr[i])
           cols.append(cc[i])
           vals.append(cv[i])
           cnt += 1
           # local nonmax suppress
           r_low = cr[i] - off if cr[i] - off >= 0 else 0
           r_high = cr[i] + off if cr[i] + off < har.shape[0] else har.shape[0]-1
           c_low = cc[i] - off if cc[i] - off >= 0 else 0
           c_high = cc[i] + off if cc[i] + off < har.shape[1] else har.shape[1]-1
           oks[r_low:r_high+1, c_low:c_high+1] = 0

   xs = np.array(rows)
   ys = np.array(cols)
   scores = np.array(vals)
   ##########################################################################
   return xs, ys, scores

"""
   FEATURE DESCRIPTOR (12 Points Implementation + 3 Points Write-up)

   Implement a SIFT-like feature descriptor by binning orientation energy
   in spatial cells surrounding an interest point.

   Unlike SIFT, you do not need to build-in rotation or scale invariance.

   A reasonable default design is to consider a 3 x 3 spatial grid consisting
   of cell of a set width (see below) surrounding an interest point, marked
   by () in the diagram below.  Using 8 orientation bins, spaced evenly in
   [-pi,pi), yields a feature vector with 3 * 3 * 8 = 72 dimensions. 

             ____ ____ ____
            |    |    |    |
            |    |    |    |
            |____|____|____|
            |    |    |    |
            |    | () |    |
            |____|____|____|
            |    |    |    |
            |    |    |    |
            |____|____|____|

                 |----|
                  width

   You will need to decide on a default spatial width.  Optionally, this can
   be a multiple of a scale factor, passed as an argument.  We will only test
   your code by calling it with scale = 1.0.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

  Arguments:
      image    - a grayscale image in the form of a 2D numpy
      xs       - numpy array of shape (N,) containing x-coordinates
      ys       - numpy array of shape (N,) containing y-coordinates
      scale    - scale factor

   Returns:
      feats    - a numpy array of shape (N,K), containing K-dimensional
                 feature descriptors at each of the N input locations
                 (using the default scheme suggested above, K = 72)
"""
def extract_features(image, xs, ys, scale = 1.0):
   # check that image is grayscale
   assert image.ndim == 2, 'image should be grayscale'
   ##########################################################################
   # TODO: YOUR CODE HERE
   N = xs.shape[0]
   K = 72
   width = 15 * scale

   mag, theta = canny_nmax(image)
   divider = np.pi / 4

   grads = np.where(mag > 0, theta // divider, np.nan)
   # # np.where operation is the same as below
   # grads = np.zeros(image.shape)
   # for r in range(image.shape[0]):
   #     for c in range(image.shape[1]):
   #         if mag[r, c] > 0:
   #             grads[r, c] = theta[r, c] // divider
   #             if grads[r, c] == 4.0:
   #                 grads[r, c] = 3.0
   #         else:
   #             grads[r, c] = np.nan

   feats = np.zeros((N, K))

   for h in range(N): # N interest pts
       r = xs[h]
       c = ys[h]

       for i in range(9): # 9 grids
           gr = i // 3 # --> 0, 1, 2
           gc = i % 3 # --> 0, 1, 2
           r_low = (r - ((width//2) + width) + width * gr).astype(int)
           r_high = (r - (width//2) + width * gr).astype(int)
           c_low = (c - ((width//2) + width) + width * gc).astype(int)
           c_high = (c - (width//2) + width * gc).astype(int)
           grid = grads[r_low:r_high, c_low:c_high]

           for j in range(8): # 8 directions
               if j == 7:
                   feats[h, i*8 + j] = np.sum(grid == 3.0) + np.sum(grid == 4.0)
               else: 
                   feats[h, i*8 + j] = np.sum(grid == -4.0 + j)
   ##########################################################################
   return feats

"""
   FEATURE MATCHING (7 Points Implementation + 3 Points Write-up)

   Given two sets of feature descriptors, extracted from two different images,
   compute the best matching feature in the second set for each feature in the
   first set.

   Matching need not be (and generally will not be) one-to-one or symmetric.
   Calling this function with the order of the feature sets swapped may
   result in different returned correspondences.

   For each match, also return a real-valued score indicating the quality of
   the match.  This score could be based on a distance ratio test, in order
   to quantify distinctiveness of the closest match in relation to the second
   closest match.  It could optionally also incorporate scores of the interest
   points at which the matched features were extracted.  You are free to
   design your own criterion. Note that you are required to implement the naive
   linear NN search. For 'lsh' and 'kdtree' search mode, you could do either to
   get full credits.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices. You are required to report the efficiency comparison
   between different modes by measure the runtime (check the benchmarking related
   codes in hw2_example.py).

   Arguments:
      feats0   - a numpy array of shape (N0, K), containing N0 K-dimensional
                 feature descriptors (generated via extract_features())
      feats1   - a numpy array of shape (N1, K), containing N1 K-dimensional
                 feature descriptors (generated via extract_features())
      scores0  - a numpy array of shape (N0,) containing the scores for the
                 interest point locations at which feats0 was extracted
                 (generated via find_interest_point())
      scores1  - a numpy array of shape (N1,) containing the scores for the
                 interest point locations at which feats1 was extracted
                 (generated via find_interest_point())
      mode     - 'naive': performs a brute force NN search

               - 'lsh': Implementing the local senstive hashing (LSH) approach
                  for fast feature matching. In LSH, the high dimensional
                  feature vectors are randomly projected into low dimension
                  space which are further binarized as boolean hashcodes. As we
                  group feature vectors by hashcodes, similar vectors may end up
                  with same 'bucket' with high propabiltiy. So that we can
                  accelerate our nearest neighbour matching through hierarchy
                  searching: first search hashcode and then find best
                  matches within the bucket.
                  Advice for impl.:
                  (1) Construct a LSH class with method like
                  compute_hash_code   (handy subroutine to project feature
                                      vector and binarize)
                  generate_hash_table (constructing hash table for all input
                                      features)
                  search_hash_table   (handy subroutine to search hash table)
                  search_feat_nn      (search nearest neighbour for input
                                       feature vector)
                  (2) It's recommended to use dictionary to maintain hashcode
                  and the associated feature vectors.
                  (3) When there is no matching for queried hashcode, find the
                  nearest hashcode as matching. When there are multiple vectors
                  with same hashcode, find the cloest one based on original
                  feature similarity.
                  (4) To improve the robustness, you can construct multiple hash tables
                  with different random project matrices and find the closest one
                  among all matched queries.
                  (5) It's recommended to fix the random seed by random.seed(0)
                  or np.random.seed(0) to make the matching behave consistenly
                  across each running.

               - 'kdtree': construct a kd-tree which will be searched in a more
                  efficient way. https://en.wikipedia.org/wiki/K-d_tree
                  Advice for impl.:
                  (1) The most important concept is to construct a KDNode. kdtree
                  is represented by its root KDNode and every node represents its
                  subtree.
                  (2) Construct a KDNode class with Variables like data (to
                  store feature points), left (reference to left node), right
                  (reference of right node) index (reference of index at original
                  point sets)and Methods like search_knn.
                  In search_knn function, you may specify a distance function,
                  input two points and returning a distance value. Distance
                  values can be any comparable type.
                  (3) You may need a user-level create function which recursively
                  creates a tree from a set of feature points. You may need specify
                  a axis on which the root-node should split to left sub-tree and
                  right sub-tree.


   Returns:
      matches  - a numpy array of shape (N0,) containing, for each feature
                 in feats0, the index of the best matching feature in feats1
      scores   - a numpy array of shape (N0,) containing a real-valued score
                 for each match
"""

class Node:
    def __init__(self, data, idx):
        self.data = data
        self.left = None
        self.right = None
        self.idx = idx

def dist(pt1, pt2):
    return np.sum(np.square(pt1 - pt2))

def insert_node(root, node):
    cd = 0
    current = root
    while True: 
        if current.data[cd] < node.data[cd]:
            if current.left:
                current = current.left
                cd = (cd+1) % 72 # K = 72
            else:
                current.left = node
                break
        else:
            if current.right:
                current = current.right
                cd = (cd+1) % 72 # K = 72
            else:
                current.right = node
                break

def create_kdtree(feats1):
    rand_idx = np.arange(1, feats1.shape[0])
    np.random.shuffle(rand_idx)
    root = Node(feats1[rand_idx[0]], rand_idx[0])
    for i in rand_idx:
        insert_node(root, Node(feats1[i], i))
    return root

def search_knn(root, target_pt):
    """
    k = 2 because we need to find d1 and d2 for the score calculation. 
    """
    prev = None
    current = root
    cd = 0
    d1 = np.inf
    d2 = np.inf
    idx1 = None
    idx2 = None

    while current:
        d = dist(current.data, target_pt)
        if d < d1:
            d2 = d1
            d1 = d
            idx2 = idx1
            idx1 = current.idx
        elif d < d2:
            d2 = d
            idx2 = current.idx

        if current.data[cd] < target_pt[cd]:
            prev = current
            current = current.left
            cd = (cd+1) % 72 # K = 72
        else:
            prev = current
            current = current.right
            cd = (cd+1) % 72 # K = 72

    return d1, d2, idx1, idx2


def match_features(feats0, feats1, scores0, scores1, mode='naive'):
   ##########################################################################
   # TODO: YOUR CODE HERE
   matches = np.zeros(feats0.shape[0]).astype(int)
   scores = np.zeros(feats0.shape[0])
   sfp = np.nextafter(0,1)

   if mode == 'naive':
       for i in range(feats0.shape[0]):
           f0 = feats0[i]
           distances = np.zeros(feats1.shape[0])
           for j in range(feats1.shape[0]):
               f1 = feats1[j]
               distances[j] = dist(f0, f1)
           matches[i] = (np.argmin(distances)).astype(int)
           d1 = np.min(distances)
           distances[matches[i]] = np.nan
           d2 = np.nanmin(distances)
           scores[i] = 1 - ((d1 + sfp) / (d2 + sfp))

   elif mode == 'kdtree': # helper functions are defined above
       kdtree = create_kdtree(feats1)
       for i in range(feats0.shape[0]):
           d1, d2, idx1, idx2 = search_knn(kdtree, feats0[i])
           matches[i] = idx1
           scores[i] = 1 - ((d1 + sfp) / (d2 + sfp))

   else:
       raise NotImplementedError
   ##########################################################################
   return matches, scores

"""
   HOUGH TRANSFORM (7 Points Implementation + 3 Points Write-up)

   Assuming two images of the same scene are related primarily by
   translational motion, use a predicted feature correspondence to
   estimate the overall translation vector t = [tx ty].

   Your implementation should use a Hough transform that tallies votes for
   translation parameters.  Each pair of matched features votes with some
   weight dependant on the confidence of the match; you may want to use your
   estimated scores to determine the weight.

   In order to accumulate votes, you will need to decide how to discretize the
   translation parameter space into bins.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      xs0     - numpy array of shape (N0,) containing x-coordinates of the
                interest points for features in the first image
      ys0     - numpy array of shape (N0,) containing y-coordinates of the
                interest points for features in the first image
      xs1     - numpy array of shape (N1,) containing x-coordinates of the
                interest points for features in the second image
      ys1     - numpy array of shape (N1,) containing y-coordinates of the
                interest points for features in the second image
      matches - a numpy array of shape (N0,) containing, for each feature in
                the first image, the index of the best match in the second
      scores  - a numpy array of shape (N0,) containing a real-valued score
                for each pair of matched features

   Returns:
      tx      - predicted translation in x-direction between images
      ty      - predicted translation in y-direction between images
      votes   - a matrix storing vote tallies; this output is provided for
                your own convenience and you are free to design its format
"""
def hough_votes(xs0, ys0, xs1, ys1, matches, scores):
   ##########################################################################
   # TODO: YOUR CODE HERE
   N = xs0.shape[0]
   XBIN_SIZE = 8
   YBIN_SIZE = 8

   max_x = (max(xs0) - min(xs1)).astype(int)
   min_x = (min(xs0) - max(xs1)).astype(int)
   max_y = (max(ys0) - min(ys1)).astype(int)
   min_y = (min(ys0) - max(ys1)).astype(int)
   txs = np.arange(min_x, max_x, XBIN_SIZE)
   tys = np.arange(min_y, max_y, YBIN_SIZE)

   votes = np.zeros((len(txs), len(tys)))

   for i in range(N):
       j = matches[i]
       x_diff = xs0[i] - xs1[j]
       y_diff = ys0[i] - ys1[j]
       x_bin = (x_diff - (min_x-XBIN_SIZE//2)) // XBIN_SIZE
       y_bin = (y_diff - (min_y-YBIN_SIZE//2)) // YBIN_SIZE
       votes[x_bin, y_bin] += scores[i]

   max_val = 0
   r = None
   c = None
   for i in range(votes.shape[0]):
       for j in range(votes.shape[1]):
           if max_val < votes[i, j]:
               max_val = votes[i, j]
               r = i
               c = j
   tx = r * XBIN_SIZE + min_x
   ty = c * YBIN_SIZE + min_y
   ##########################################################################
   return tx, ty, votes
