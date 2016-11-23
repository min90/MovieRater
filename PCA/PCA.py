import Persistence.Reader as rd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
from matplotlib.mlab import PCA as mlabPCA

###############################################################################
###
###     Parameters
###
###############################################################################

# Normalize data ?
normalize = True
standardize = False

###############################################################################
###
###     Dataset reader
###
###############################################################################

reader = rd.CSVReader
data = reader.read("../cleaned_data.csv")
nRows = len(data)

###############################################################################
###
###     Dataset array creation
###
###############################################################################

Dataset = []

# count number of points for each class
ptsInClasses = [0,0,0,0,0,0,0,0,0,0]

# color
classColors  = [None] * 10
classColors[0] = '#FFFFFF'
classColors[1] = '#FFFF00'
classColors[2] = '#FFDD00'
classColors[3] = '#FFBB00'
classColors[4] = '#FF9900'
classColors[5] = '#FF7700'
classColors[6] = '#FF5500'
classColors[7] = '#FF3300'
classColors[8] = '#FF1100'
classColors[9] = '#B20000'

# create the dataset used for PCA
for row in data:
    dirLikes = int(row[5])
    act1Likes = int(row[6])
    act2Likes = int(row[7])
    act3Likes = int(row[8])

    r = float(row[4])
    if (r < 1)  : ptsInClasses[0] += 1
    elif (r < 2): ptsInClasses[1] += 1
    elif (r < 3): ptsInClasses[2] += 1
    elif (r < 4): ptsInClasses[3] += 1
    elif (r < 5): ptsInClasses[4] += 1
    elif (r < 6): ptsInClasses[5] += 1
    elif (r < 7): ptsInClasses[6] += 1
    elif (r < 8): ptsInClasses[7] += 1
    elif (r < 9): ptsInClasses[8] += 1
    else        : ptsInClasses[9] += 1

    row = [dirLikes, act1Likes, act2Likes, act3Likes, r]
    Dataset.append(row)

# sort dataset by ascending rating
Dataset.sort(key=lambda x: x[4])

# make numpy array
Dataset = np.array(Dataset)
Dts = Dataset[:,0:4].T

# some information
print("\n~~~~~ Dataset ~~~~~")
print("Dataset shape : " + str(Dts.shape))
print("Distribution (rating):", ptsInClasses)

###############################################################################
###
###     Normalization
###
###############################################################################

# get all the like values
all_likes = []
for row in data:
    all_likes.append(int(row[5]))
    all_likes.append(int(row[6]))
    all_likes.append(int(row[7]))
    all_likes.append(int(row[8]))

# get min/max
min = min(all_likes)
max = max(all_likes)

# mean & standard deviation
mean = np.mean(all_likes)
std = np.std(all_likes)

if normalize:
    # rescale values
    for i in range(nRows):
        Dts[0][i] = (Dts[0][i] - min) / (max - min)
        Dts[1][i] = (Dts[1][i] - min) / (max - min)
        Dts[2][i] = (Dts[2][i] - min) / (max - min)
        Dts[3][i] = (Dts[3][i] - min) / (max - min)
    print('Data normalized. (min-max)')

elif standardize:
    # rescale values
    for i in range(nRows):
        Dts[0][i] = (Dts[0][i] - mean) / (std)
        Dts[1][i] = (Dts[1][i] - mean) / (std)
        Dts[2][i] = (Dts[2][i] - mean) / (std)
        Dts[3][i] = (Dts[3][i] - mean) / (std)
    print('Data standardized. (z-score)')

###############################################################################
###
###     Manual PCA
###
###############################################################################

print("\nManual PCA.")

# ===============================================
# Computing the d-dimensional mean vector

print('\n~~~~~ Mean vector ~~~~~')

mean_dir = np.mean(Dts[0, :])
mean_act1 = np.mean(Dts[1, :])
mean_act2 = np.mean(Dts[2, :])
mean_act3 = np.mean(Dts[3, :])
mean_vector = np.array( [ [mean_dir], [mean_act1], [mean_act2], [mean_act3] ] )

print('DirectorLikes: \tMin: ' + str(np.min(Dts[0, :])) + '\tMax: ' + str(np.max(Dts[0, :])) + '\tStandard deviation: ' + str(np.std(Dts[0, :])) + '\tMean: ' + str(mean_dir))
print('Actor-1-Likes: \tMin: ' + str(np.min(Dts[1, :])) + '\tMax: ' + str(np.max(Dts[1, :])) + '\tStandard deviation: ' + str(np.std(Dts[1, :])) + '\tMean: ' + str(mean_act1))
print('Actor-2-Likes: \tMin: ' + str(np.min(Dts[2, :])) + '\tMax: ' + str(np.max(Dts[2, :])) + '\tStandard deviation: ' + str(np.std(Dts[2, :])) + '\tMean: ' + str(mean_act2))
print('Actor-3-Likes: \tMin: ' + str(np.min(Dts[3, :])) + '\tMax: ' + str(np.max(Dts[3, :])) + '\tStandard deviation: ' + str(np.std(Dts[3, :])) + '\tMean: ' + str(mean_act3))
print('Mean vector generated.')

# ===============================================
# Computing the Scatter Matrix

print('\n~~~~~ Scatter matrix ~~~~~')

scatter_matrix = np.zeros((4, 4))

for i in range(Dts.shape[1]):
    scatter_matrix += (Dts[:,i].reshape(4,1) - mean_vector).dot((Dts[:,i].reshape(4,1) - mean_vector).T)

print('Scatter matrix generated.')
print(scatter_matrix)

# ===============================================
# Computing the Covariance Matrix

print('\n~~~~~ Covariance matrix ~~~~~')

cov_mat = np.cov( [ Dts[0,:], Dts[1,:], Dts[2,:], Dts[3,:] ] )
print('Covariance matrix generated.')
print(cov_mat)

# ===============================================
# Computing eigenvectors and corresponding eigenvalues

print('\n~~~~~ Eigen vectors & Eigen values ~~~~~')

# eigenvectors and eigenvalues from the scatter matrix
eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

# eigenvectors and eigenvalues from the covariance matrix
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

for i in range(len(eig_val_sc)):
    eigvec_sc = eig_vec_sc[:,i].reshape(1,4).T
    eigvec_cov = eig_vec_cov[:,i].reshape(1,4).T
    assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'

    print(30*"*")
    print('\tEigenvector {} : \n{}'.format(i+1, eigvec_sc))
    print('\tEigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))
    print('\tEigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
    print('\tScaling factor: ', eig_val_sc[i]/eig_val_cov[i])

# ===============================================
# Checking eigen vectors/values computation

print('\n~~~~~ Eigen vectors/values computation check ~~~~~')

for i in range(len(eig_val_cov)):
    eigv = eig_vec_cov[:,i].reshape(1,4).T
    np.testing.assert_array_almost_equal(cov_mat.dot(eigv), eig_val_cov[i] * eigv, decimal=6, err_msg='ERROR', verbose=True)
    print('Eigen vector/value #{} OK.'.format(i+1))

print('Checking done.')

# ===============================================
# Sorting the eigenvectors by decreasing eigenvalues

print('\n~~~~~ Eigen vectors sorting ~~~~~')

# Just check if the eigen vectors have all the same unit length (1)
for ev in eig_vec_sc:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

print('Vectors are now sorted. Let\'s check with the eigenvalues order :')

# Visually check that the list is correctly sorted by decreasing eigenvalues
for pair in eig_pairs:
    print(pair[0])

# ===============================================
# Ploting cumulative sum of explained variance of each PC

print('\n~~~~~ How many PCs do we need ? ~~~~~')

PC = []
PC.append([0, 0])
sum = 0
sum_eigenvalues = np.sum(eig_val_cov)

print('Eigenvalue sum : ' + str(sum_eigenvalues))

for k in range(len(eig_val_cov)):
    sum += (eig_val_cov[k] / sum_eigenvalues)*100
    PC.append([k+1, sum])

PC = np.array(PC)
print('Cumulative sum of explained variances : (%)')
print(PC)

plt.plot(PC[:,0], PC[:,1], '-o', markersize=3, color='blue', alpha=1)
plt.xlabel('Number of PCs')
plt.ylabel('Cumulative variance (%)')
plt.xlim([0, 4])
plt.ylim([0, 100])
plt.xticks(np.arange(0, 5, 1))
plt.title('Amount of variance explained by PCs')
print("Graph displayed.")
plt.show()

# ===============================================
# Choosing 1 eigenvector with the largest eigenvalue

print('\n~~~~~ Transformation matrix ~~~~~')
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
print('Matrix W : (' + str(matrix_w.shape) + ')\n', matrix_w)

# ===============================================
# Transforming the samples onto the new subspace

print('\n~~~~~ Data transformation ~~~~~')
transformed = matrix_w.T.dot(Dts)
print('Data transformed. (' + str(transformed.shape) + ')\n', transformed)
print('Done.')

# ===============================================
# Final plot

print('\n~~~~~ Final plot Feature/Feature ~~~~~')

# plot each class
for c in range(len(ptsInClasses)):
    if (ptsInClasses[c] > 0):
        begin = np.sum(ptsInClasses[0:c])
        end = begin + ptsInClasses[c]
        print('Class #' + str(c) + ' (rating ' + str(c) + '-' + str(c+1) + ') : ' + str(begin) + ':' + str(end) + ' => ' + str(ptsInClasses[c]) + ' points')
        plt.plot(transformed[0, begin:end], transformed[1, begin:end], 'o', markersize=6, color=classColors[c], alpha=0.9, label='rating '+str(c)+'-'+str(c+1))

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

if normalize:
    plt.xlim([0, 1])
    plt.ylim([0, 1])
else:
    plt.xlim([-10000, 700000])
    plt.ylim([-75000, 125000])

plt.legend()
plt.title('PCA work visualization')
print("Graph displayed.")
plt.show()