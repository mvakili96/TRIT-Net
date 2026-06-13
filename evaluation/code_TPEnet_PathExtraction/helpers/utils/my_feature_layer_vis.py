import cv2

def visualize_featuremap(x):
    print(x.shape)
    x_numpy = x.detach().cpu().numpy()
    for i,layer in enumerate(x_numpy[0]):
        cv2.imshow("A" + str(i),layer)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
