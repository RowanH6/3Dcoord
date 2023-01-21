import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# left camera intrinsics

cL_mat = [[1387.500156704989, 0.0, 637.575611719055],
          [0.0, 1387.3291379610362, 342.867127287223],
          [0.0, 0.0, 1.0]]

# right camera intrinsics

cR_mat = [[1652.6476793663112, 0.0, 665.7071649002949],
          [0.0, 1652.2174938724572, 338.4559034965347],
          [0.0, 0.0, 1.0]]

# focal lengths from left camera (f_u -> x, f_v -> y)

f_u = cL_mat[0][0]
f_v = cL_mat[1][1]

# distance between cameras (m)

b = 0.253

# center coordinates of left camera (c_u -> x, c_v -> y)

c_u = 320
c_v = 240

# pixel disparity

def disp(x1, x2):
    return abs(x1 - x2)

# 3D coordinate calculation

def pt(pL, pR):
    u_l, v_l = pL[0], pL[1]
    u_r, v_r = pR[0], pR[1]

    d = disp(u_l, u_r)

    x = (b/d) * (u_l - c_u)
    y = (b/d) * (f_u/f_v) * (v_l - c_v)
    z = (b/d) * (f_u)

    return [x, y, z]

# chessboard point detection

right = cv.imread('3Dcoord/capture/r3.jpg')
left = cv.imread('3Dcoord/capture/l3.jpg')

g_l = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
g_r = cv.cvtColor(right, cv.COLOR_BGR2GRAY)

chessboard_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE

columns = 7
rows = 4

lret, l_corners = cv.findChessboardCorners(g_l, (columns, rows), chessboard_flags)
rret, r_corners = cv.findChessboardCorners(g_r, (columns, rows), chessboard_flags)

l_corners, r_corners = np.double(l_corners), np.int32(r_corners)
l_corners, r_corners = l_corners[:, 0, :], r_corners[:, 0, :]

# 3D point calculation

pts_3D = []

for p in range (len(l_corners[:, 0])):
    pt_3D = pt(l_corners[p, :], r_corners[p, :])
    pts_3D.append(pt_3D)

pts_3D = np.double(pts_3D)

# plot on x, y, z axes

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(pts_3D[:, 0], pts_3D[:, 1], pts_3D[:, 2], 'gray')
ax.set_zlim(0, max(pts_3D[:, 2]) + 0.2)
ax.set_ylim(min(pts_3D[:, 1]) - 0.1, max(pts_3D[:, 1]) + 0.1)
ax.set_xlim(min(pts_3D[:, 0]) - 0.1, max(pts_3D[:, 0]) + 0.1)

plt.show()