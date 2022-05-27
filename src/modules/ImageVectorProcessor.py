from matplotlib import pyplot as plt

def chopImage(image, N):
    return [y for x in [image[:N], [float(0) for i in range(len(image)-N)]] for y in x]

def show_images(images,N,M) -> None:
    n: int = len(images)
    f = plt.figure()

    for i in range(n):
        f.add_subplot(N, M, i + 1)
        plt.imshow(images[i])

    plt.show(block=True)
def show_errors(error_list):
    n: int = len(error_list)
    
    f = plt.figure()
    for i in range(n):
        f.add_subplot(4, 1, i + 1)
        plt.plot(np.log(error_list[i]))
        plt.ylabel('Error')

    plt.show(block=True)
