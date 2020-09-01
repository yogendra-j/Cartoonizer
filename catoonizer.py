import cv2
import numpy as np
from collections import defaultdict
from scipy import stats


def makecartoon(img):
    '''
    arg: img = image
    return catoonified image

    useses bilaterral blur, canny edge detection, k-means clustering
    '''
    if max(img.shape) > 2000:
        scale = 2000 / max(img.shape)
        img = cv2.resize(
            img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    kernal = np.ones((3, 3), np.uint8)
    out = blurring(img,)
    img = cv2.medianBlur(img, 1)
    edge = edge_detection(out)

    out = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hists = histogram(out)
    cluster = []
    for hist in hists:
        cluster.append(find_K(hist))
    # print(hists)
    h, w, channels = img.shape
    out = out.reshape((-1, channels))
    # print(cluster)
    for i in range(channels):
        single_channel = out[:, i]
        indx = np.argmin(
            np.abs(single_channel[:, np.newaxis] - cluster[i]), axis=1)
        out[:, i] = cluster[i][indx]
    out = out.reshape((h, w, channels))
    out = cv2.cvtColor(out, cv2.COLOR_HSV2RGB)
    draw_contours(out, edge)
    erode(out, kernal)
    return out


def blurring(output, dia=3, sigma_c=200, sigma_s=200):
    '''takes image as inut and return blured image using bilateral filter'''
    h, w, c = output.shape
    for i in range(c):
        output[:, :, i] = cv2.bilateralFilter(
            output[:, :, i], dia, sigma_c, sigma_s)
    return output


def edge_detection(img, thr1=115, thr2=185):
    '''takes image and returns output from canny edge detection algo'''

    return cv2.Canny(img, thr1, thr2)


def histogram(img):
    '''takes image and returns histograms of hsv imaget'''

    hists = []

    hist, _ = np.histogram(img[:, :, 0], bins=np.arange(180 + 1))
    hists.append(hist)
    hist, _ = np.histogram(img[:, :, 1], bins=np.arange(256 + 1))
    hists.append(hist)
    hist, _ = np.histogram(img[:, :, 2], bins=np.arange(256 + 1))
    hists.append(hist)
    return hists


def update_centroid(Clusters, hist):
    '''optimises clusters'''
    while True:
        groups = defaultdict(list)

        for i, h in enumerate(hist):
            if h == 0:
                continue  # no point for this value

            diff = np.abs(Clusters - i)
            # cal diff from each cluster and
            # assign to nearest
            idx = np.argmin(diff)
            # idx is index of cluster
            groups[idx].append(i)

        Clusters_new = np.array(Clusters)

        for i, indx in groups.items():
            if np.sum(hist[indx]) == 0:
                continue
            # hist values are freqencies
            Clusters_new[i] = int(
                np.sum(indx * hist[indx]) / np.sum(hist[indx]))

        if np.sum(Clusters_new - Clusters) == 0:
            break
        Clusters = Clusters_new
    return Clusters, groups


def find_K(hist):
    '''findes optimal number of clusters'''
    alpha = 1
    N = 70
    clusters = np.array([128])

    while True:
        clusters, groups = update_centroid(clusters, hist)

        clusters_new = set()
        for i, indx in groups.items():
            if len(indx) < N:  # too less points in the cluster
                clusters_new.add(clusters[i])
                continue

            z, pval = stats.normaltest(hist[indx])
            if (pval < alpha):  # not enough variation
                left = 0 if i == 0 else clusters[i - 1]
                right = len(hist) - 1 if i == len(clusters) - \
                    1 else clusters[i + 1]
                delta = right - left
                if delta >= 3:
                    c1 = (clusters[i] + left) / 2
                    c2 = (clusters[i] + right) / 2
                    clusters_new.update([c1, c2])
                else:
                    clusters_new.add(clusters[i])
            else:
                clusters_new.add(clusters[i])

        if len(clusters_new) == len(clusters):
            break
        else:
            clusters = np.array(sorted(clusters_new))

    return clusters


def draw_contours(img, edge):
    countours, _ = cv2.findContours(
        edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, countours, -1, 0, thickness=1)


def erode(img, kernal):
    return cv2.erode(img, kernal, iterations=1)


if __name__ == "__main__":
    img = cv2.imread("IMG-20200121-WA0021.jpg")
    out = makecartoon(img)
    cv2.imshow("output", out)
    img = cv2.imread("20191211_004108.jpg")
    out = makecartoon(img)
    cv2.imshow("output2", out)
    img = cv2.imread("20200130_201459.jpg")
    out = makecartoon(img)
    cv2.imshow("output3", out)
    img = cv2.imread("me.jpeg")
    img = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))
    out = makecartoon(img)
    cv2.imshow("output4", out)
    # cap = cv2.VideoCapture(0)
    # while True:
    #     _, img = cap.read()
    #     out = makecartoon(img)
    #     cv2.imshow("output5", out)
    #     if cv2.waitKey(2) & 0xFF == ord('q'):
    #         break
    # cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
