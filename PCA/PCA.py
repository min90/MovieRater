import Persistence.Reader as rd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
from matplotlib.mlab import PCA as mlabPCA

###############################################################################
###
###     Method to use
###
###############################################################################

# PCA with SKLearn library
performSKLearnPCA = False

# PCA with MLAB library
performMlabPCA = False

# Manual PCA
performManualPCA = True

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

# create the dataset used for PCA
for i in range(0, nRows):
    dirLikes = int(data[i][5])
    act1Likes = int(data[i][6])
    act2Likes = int(data[i][7])
    act3Likes = int(data[i][8])
    row = [dirLikes, act1Likes, act2Likes, act3Likes]
    Dataset.append(row)

Dataset = np.array(Dataset)
Dts = Dataset.T

print("\n~~~~~ Dataset ~~~~~")
print("Dts shape : " + str(Dts.shape))

###############################################################################
###
###     PCA SKLEARN
###
###############################################################################

if performSKLearnPCA == True:
    print("\n~~~~~ SKLearn PCA ~~~~~")

    # perfom Sklearn PCA
    sklearn_pca = sklearnPCA()
    sklearn_transf = sklearn_pca.fit_transform(Dataset)

    # RATING / DIRECTOR
    plt.plot(sklearn_transf[0:nRows,0],sklearn_transf[0:nRows,4], 'o', markersize=2, color='red', alpha=0.5, label='whole dataset')
    plt.xlabel('DIRECTOR')
    plt.ylabel('RATING')
    plt.xlim([-4000,8000])
    plt.ylim([-10,10])
    plt.legend()
    plt.title('SKLEARN - Rating/Director')
    print("Graph Rating/Director displayed.")
    plt.show()

    # RATING / ACTOR 1
    plt.plot(sklearn_transf[0:nRows,1],sklearn_transf[0:nRows,4], 'o', markersize=3, color='red', alpha=0.5, label='whole dataset')
    plt.xlabel('ACTOR 1')
    plt.ylabel('RATING')
    plt.xlim([-5000,5000])
    plt.ylim([-8,8])
    plt.legend()
    plt.title('SKLEARN - Rating/Actor 1')
    print("Graph Rating/Actor 1 displayed.")
    plt.show()

    # RATING / ACTOR 2
    plt.plot(sklearn_transf[0:nRows,2],sklearn_transf[0:nRows,4], 'o', markersize=3, color='red', alpha=0.5, label='whole dataset')
    plt.xlabel('ACTOR 2')
    plt.ylabel('RATING')
    plt.xlim([-5000,5000])
    plt.ylim([-5,8])
    plt.legend()
    plt.title('SKLEARN - Rating/Actor 2')
    print("Graph Rating/Actor 2 displayed.")
    plt.show()

    # RATING / ACTOR 3
    plt.plot(sklearn_transf[0:nRows,3],sklearn_transf[0:nRows,4], 'o', markersize=3, color='red', alpha=0.5, label='whole dataset')
    plt.xlabel('ACTOR 3')
    plt.ylabel('RATING')
    plt.xlim([-2500,2500])
    plt.ylim([-5,8])
    plt.legend()
    plt.title('SKLEARN - Rating/Actor 3')
    print("Graph Rating/Actor 3 displayed.")
    plt.show()

###############################################################################
###
###     PCA MLAB
###
###############################################################################

if performMlabPCA == True:
    print("\n~~~~~ MLAB PCA ~~~~~")

    # perform Mlab PCA
    mlab_pca = mlabPCA(Dataset)

    print("Results :")
    print("\tnumber of rows : " + str(mlab_pca.numrows))
    print("\tnumber of cols : " + str(mlab_pca.numcols))
    print("\tmeans of a : ")
    print(mlab_pca.mu)
    print("\tstandard deviations of a : ")
    print(mlab_pca.sigma)
    print("\tproportion of variance of each of the principal components : ")
    print(mlab_pca.fracs)
    print('\tPC axes in terms of the measurement axes scaled by the standard deviations:\n', mlab_pca.Wt)

    # RATING / DIRECTOR
    plt.plot(mlab_pca.Y[0:nRows,0], mlab_pca.Y[0:nRows,4], 'o', markersize=2, color='blue', alpha=0.5, label='whole dataset')
    plt.xlabel('DIRECTOR')
    plt.ylabel('RATING')
    plt.xlim([-3,6])
    plt.ylim([-4,5])
    plt.legend()
    plt.title('Matplotlib.MLAB - Rating/Director')
    plt.show()

    # RATING / ACTOR 1
    plt.plot(mlab_pca.Y[0:nRows,1], mlab_pca.Y[0:nRows,4], 'o', markersize=2, color='blue', alpha=0.5, label='whole dataset')
    plt.xlabel('ACTOR 1')
    plt.ylabel('RATING')
    plt.xlim([-3,6])
    plt.ylim([-4,5])
    plt.legend()
    plt.title('Matplotlib.MLAB - Rating/Actor 1')
    plt.show()

    # RATING / ACTOR 2
    plt.plot(mlab_pca.Y[0:nRows,2], mlab_pca.Y[0:nRows,4], 'o', markersize=2, color='blue', alpha=0.5, label='whole dataset')
    plt.xlabel('ACTOR 2')
    plt.ylabel('RATING')
    plt.xlim([-3,6])
    plt.ylim([-4,5])
    plt.legend()
    plt.title('Matplotlib.MLAB - Rating/Actor 2')
    plt.show()

    # RATING / ACTOR 3
    plt.plot(mlab_pca.Y[0:nRows,3], mlab_pca.Y[0:nRows,4], 'o', markersize=2, color='blue', alpha=0.5, label='whole dataset')
    plt.xlabel('ACTOR 3')
    plt.ylabel('RATING')
    plt.xlim([-3,6])
    plt.ylim([-4,5])
    plt.legend()
    plt.title('Matplotlib.MLAB - Rating/Actor 3')
    plt.show()

###############################################################################
###
###     Manual PCA
###
###############################################################################

if performManualPCA == True:
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

    plt.plot(PC[:,0], PC[:,1], '-o', markersize=3, color='blue', alpha=1, label='whole dataset')
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
    plt.plot(transformed[0,:], transformed[1,:], '-o', markersize=3, color='green', alpha=1, label='whole dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim([-10000, 700000])
    plt.ylim([-75000, 125000])
    plt.title('PCA work visualization')
    print("Graph displayed.")
    plt.show()