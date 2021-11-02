import cv2
import numpy as np

rgb = cv2.imread('./test.bmp',cv2.IMREAD_COLOR)
ir = cv2.imread('./test_ir.bmp',cv2.IMREAD_COLOR)



cv2.imshow('ir0',rgb[:,:,0])
cv2.imshow('ir1',rgb[:,:,1])
cv2.imshow('ir2',rgb[:,:,2])
cv2.waitKey(0)
cv2.destroyAllWindows()