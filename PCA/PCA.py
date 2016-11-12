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
performSKLearnPCA = True

# PCA with MLAB library
performMlabPCA = False

# Manual PCA
performManualPCA = False

###############################################################################
###
###     Dataset reader
###
###############################################################################

reader = rd.CSVReader
data, nb_directors, nb_actors = reader.read("../cleaned_data.csv")
nRows = len(data)

###############################################################################
###
###     Dataset array creation
###
###############################################################################

# get the vector for an actor
def actorVector(pos):
    return int(pos) # temporary

    vector = np.zeros(nb_actors)
    vector[int(pos)] = 1
    return vector

# get the vector for a director
def directorVector(pos):
    return int(pos) # temporary

    vector = np.zeros(nb_directors)
    vector[int(pos)] = 1
    return vector

Dataset = []

# create the dataset used for PCA
for i in range(0, nRows):
    dir = directorVector(data[i][4])
    act1 = actorVector(data[i][5])
    act2 = actorVector(data[i][6])
    act3 = actorVector(data[i][7])
    row = [dir, act1, act2, act3, float(data[i][8])]
    Dataset.append(row)

Dataset = np.array(Dataset)

print("\n~~~~~ Dataset ~~~~~")
print("Dataset shape : " + str(Dataset.shape))

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
    print("\tnumdims array of means of a : ")
    print(mlab_pca.mu)
    print("\tnumdims array of standard deviation of a : ")
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
    print("\n~~~~~ Manual PCA ~~~~~")

    # ===============================================
    # Computing the d-dimensional mean vector

    mean_dir = np.mean(Dataset[:, 0])
    mean_act1 = np.mean(Dataset[:, 1])
    mean_act2 = np.mean(Dataset[:, 2])
    mean_act3 = np.mean(Dataset[:, 3])
    mean_rating = np.mean(Dataset[:, 4])

    mean_vector = np.array( [ [mean_dir], [mean_act1], [mean_act2], [mean_act3], [mean_rating] ] )

    print('Mean Vector :\n', mean_vector)

    # ===============================================
    # Computing the Scatter Matrix

    scatter_matrix = np.zeros((5, 5))

    for i in range(Dataset.shape[0]):
        scatter_matrix += (Dataset[i,:].reshape(1,5) - mean_vector).dot((Dataset[i,:].reshape(1,5) - mean_vector))

    print('Scatter Matrix :\n', scatter_matrix)

    # ===============================================
    # Computing the Covariance Matrix

    cov_mat = np.cov([Dataset[:,0], Dataset[:,1], Dataset[:,2], Dataset[:,3], Dataset[:,4]])
    print('Covariance Matrix :\n', cov_mat)

    # ===============================================
    # Computing eigenvectors and corresponding eigenvalues

    print('\n' + 50 * '=')
    print("Eigenvectors & Eigenvalues\n")

    # eigenvectors and eigenvalues from the scatter matrix
    eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

    # eigenvectors and eigenvalues from the covariance matrix
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

    for i in range(len(eig_val_sc)):
        eigvec_sc = eig_vec_sc[i,:].reshape(5,1)
        eigvec_cov = eig_vec_cov[i,:].reshape(5,1)
        assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'

        print('\tEigenvector {} : \n{}'.format(i+1, eigvec_sc))
        print('\tEigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))
        print('\tEigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
        print('\tScaling factor: ', eig_val_sc[i]/eig_val_cov[i])
        print('\t' + 40 * '-')

    # ===============================================
    # Checking eigen vectors/values computation

    for i in range(len(eig_val_sc)):
        eigv = eig_vec_sc[i,:].reshape(5,1)
        np.testing.assert_array_almost_equal(scatter_matrix.dot(eigv), eig_val_sc[i] * eigv, decimal=6, err_msg='', verbose=True)

    # ===============================================
    # Sorting the eigenvectors by decreasing eigenvalues

    for ev in eig_vec_sc:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[i,:]) for i in range(len(eig_val_sc))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    for i in eig_pairs:
        print(i[0])

    # ===============================================
    # Choosing k eigenvectors with the largest eigenvalues

    #matrix_w = np.hstack((eig_pairs[0][1].reshape(5,1), eig_pairs[1][1].reshape(5,1)))
    #print('Matrix W :\n', matrix_w)

    # Transforming the samples onto the new subspace
    #transformed = matrix_w.dot(Dataset)
    #print(matrix_w.shape)