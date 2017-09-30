import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import sys
from pylibdmtx import pylibdmtx

# Thresholding
THR_MAX_VAL = 255
THR_BLOCK_SIZE = 61
THR_OFFSET = 8
THR_BLUR_WINDOW = (6, 6)
P__THR_MAX_VAL = 255
P_THR_BLOCK_SIZE = 31
P_THR_OFFSET = 8
P_THR_BLUR_WINDOW = (3, 3)
TOLERANCE_FACTOR = 0.002

# Paper detection
AREA_THR = 0.02

# Paper orientation
REF_PT_RATIO = 1.053
REF_PT_RANGE = 0.012

# Transformation
MATCHING_CORNERS = np.float32([[578, 783], [33, 783], [33, 160], [578, 160]])
PAPER_SIZE = (662, 942)

# Grid trimming
OFFSET = 8
COL_WIDTH = 136.25
ROW_HEIGHT = 41.53
ROW_BASE = 160
COL_BASE = 33
NUMBER_COL_WIDTH = 30
TRIM_ROWS_TO_SCAN = 16
TRIM_THRESHOLD = 0.85

# Scanning
LEFT_RIGHT_MARGIN = 8
TOP_BOTTOM_MARGIN = 10

# QR code
QR_ROW_MIN = 0
QR_ROW_MAX = 113
QR_COL_MIN = 522
QR_COL_MAX = 636
QR_TIMEOUT = 5


def draw_polygon(plt, A, B, C, D, style='r-'):
    plt.plot([e[0] for e in [A, B, C, D, A]], [e[1] for e in [A, B, C, D, A]], style)


class RawPhoto:
    def __init__(self, img_path):
        self.raw_img = cv2.imread(img_path, 0)
        blr_img = cv2.blur(self.raw_img, THR_BLUR_WINDOW)
        self.thr_img = cv2.adaptiveThreshold(blr_img, THR_MAX_VAL,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY,
                                             THR_BLOCK_SIZE, THR_OFFSET)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self.morph_img = cv2.morphologyEx(self.thr_img, cv2.MORPH_CLOSE, kernel)
        print 'Shape:', self.raw_img.shape
        plt.subplot(1, 2, 1)
        plt.imshow(self.morph_img)
        plt.subplot(1, 2, 2)
        plt.imshow(self.raw_img)

    def search_contours(self, num_papers):
        _, contours, _ = cv2.findContours(self.morph_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        print 'num contours', len(contours)
        all_approximations = []
        tolerance = int(TOLERANCE_FACTOR * np.sum(self.raw_img.shape[:2]))
        for i, ct in enumerate(contours):
            approx = cv2.approxPolyDP(ct, tolerance, True)  # True for approx_curve
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            all_approximations.append((approx, cv2.contourArea(approx)))
        print 'num concave quadrangles', len(all_approximations)
        fun_sort_by_area = lambda entry: entry[1]
        approximations = sorted(all_approximations, 
                                key=fun_sort_by_area, reverse=True)[:num_papers]
        count = 0
        for approx, area in all_approximations:
            if area < approximations[0][1] * AREA_THR:
                break
            count += 1
        approximations = approximations[:count]
        print approximations
        #num_dropped = num_papers - count
        #if num_dropped > 0:
        #    print '[WARNING] %d papers dropped.' % num_dropped

        for approx, area in all_approximations[:100]:
            A = approx[0][0]
            B = approx[1][0]
            C = approx[2][0]
            D = approx[3][0]
            draw_polygon(plt, A, B, C, D)
        plt.show()

        print approximations
        
        self.approximations = approximations
        #self.num_dropped = num_dropped

    def extract_papers(self):
        self.papers = []
        for approx, _ in self.approximations:
            try:
                ordered_corners = self.find_orientation(approx)
            except IndexError:
                continue
            trans_matrix = cv2.getPerspectiveTransform(ordered_corners, MATCHING_CORNERS)
            paper = cv2.warpPerspective(self.raw_img, trans_matrix, PAPER_SIZE)
            print 'perspective transform done'
            #plt.subplot(1, 2, 2)
            #plt.imshow(paper)
            #self.papers.append(Paper(paper))

    def find_orientation(self, approx):
        """
        Orientate a paper region based on the black bock besides the left edge.
        :param approx: vertices of the paper rectangle
        :return: transformed vertices of the paper rectangle
        """
        # Define corner points
        A = approx[0][0]
        B = approx[1][0]
        C = approx[2][0]
        D = approx[3][0]

        # Define center points of edges
        E = (int((A[0] + B[0]) / 2), int((A[1] + B[1]) / 2))
        F = (int((B[0] + C[0]) / 2), int((B[1] + C[1]) / 2))
        G = (int((C[0] + D[0]) / 2), int((C[1] + D[1]) / 2))
        H = (int((D[0] + A[0]) / 2), int((D[1] + A[1]) / 2))

        # Define ref points (points just a bit outside the mid points)
        I = (int(G[0] + REF_PT_RATIO * (E[0] - G[0])),
             int(G[1] + REF_PT_RATIO * (E[1] - G[1])))
        J = (int(H[0] + REF_PT_RATIO * (F[0] - H[0])),
             int(H[1] + REF_PT_RATIO * (F[1] - H[1])))
        K = (int(E[0] + REF_PT_RATIO * (G[0] - E[0])),
             int(E[1] + REF_PT_RATIO * (G[1] - E[1])))
        L = (int(F[0] + REF_PT_RATIO * (H[0] - F[0])),
             int(F[1] + REF_PT_RATIO * (H[1] - F[1])))

        #draw_polygon(plt, I, J, K, L, 'bx')

        # Check brightnesses
        brightnesses = []
        offset = int(REF_PT_RANGE * ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2) ** 0.5)
        for pt in [I, J, K, L]:
            i_base = pt[0] - offset
            j_base = pt[1] - offset
            brightness = 0
            for i in range(offset * 2):
                for j in range(offset * 2):
                    brightness += self.raw_img[j_base + j][i_base + i]
            brightnesses.append(brightness)

        # Orientate rectangle
        r_id = brightnesses.index(min(brightnesses))
        transform = {0: B, 1: C, 2: D, 3: A}
        ordered = np.float32([transform[nid].tolist() for nid in 
                                [(r_id + offset) % 4 for offset in range(4)]])
        return ordered

    def process_papers(self):
        good_papers = []
        for paper in self.papers:
            paper.read_datamatrix()
            if not paper.test_id:
                continue
            paper.trim_grids()
            paper.read_all_answers()
            good_papers.append(paper)
        self.papers = good_papers


    def read_content(self, num_papers):
        self.search_contours(num_papers * 3)
        self.extract_papers()
        self.process_papers()
        dict = {
            'num_missed': num_papers - len(self.papers),
            'res': [{
                    'test_id': p.test_id,
                    'paper_id': p.paper_id,
                    'marked_ans': p.marked_ans
                } for p in self.papers]
        }
        return dict


class Paper:
    def __init__(self, raw_img, num_questions=60):
        self.raw_img = raw_img
        blr_img = cv2.blur(self.raw_img, P_THR_BLUR_WINDOW)
        self.thr_img = cv2.adaptiveThreshold(blr_img, THR_MAX_VAL,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY,
                                             P_THR_BLOCK_SIZE, P_THR_OFFSET)
        self.num_questions = num_questions

    def trim_grids(self):
        self.answer_imgs = []
        for j in range(4):
            left_edge = COL_BASE + int(j * COL_WIDTH) - OFFSET + NUMBER_COL_WIDTH
            right_edge = COL_BASE + int((j + 1) * COL_WIDTH) + OFFSET
            for i in range(15):
                top_edge = ROW_BASE + int(i * ROW_HEIGHT) - OFFSET
                bottom_edge = ROW_BASE + int((i + 1) * ROW_HEIGHT) + OFFSET
                untrimed_grid_raw = self.raw_img[top_edge:bottom_edge, left_edge:right_edge]
                untrimed_grid_thr = self.thr_img[top_edge:bottom_edge, left_edge:right_edge]
                trimmed_grid_raw, trimmed_grid_thr = \
                    self.remove_edges(untrimed_grid_raw, untrimed_grid_thr)
                self.answer_imgs.append(trimmed_grid_raw)
        assert len(self.answer_imgs) == 60
        

    def remove_edges(self, ans_img_raw, ans_img_thr):
        """
        Trims the edges of each block by traversing lines inverse until we _hit and pass_ 
        a black line.
        Four sides are rather repetitive and could be refactored.
        :param ans_img_raw: raw image of the answer block
        :param ans_img_thr: binary image of the answer block
        :return: trimmed image of the answer block, raw and binary (we need the raw picture 
        for reading the answers)
        """
        h, w = ans_img_thr.shape[:2]

        # Top
        t = 0
        flag = False
        for t in range(TRIM_ROWS_TO_SCAN):
            if sum(ans_img_thr[t]) > TRIM_THRESHOLD * w * 255:
                flag = True
            else:
                if flag:
                    break
        if t == TRIM_ROWS_TO_SCAN - 1:
            t = 0

        # Bottom
        b = 0
        flag = False
        for b in range(TRIM_ROWS_TO_SCAN):
            if sum(ans_img_thr[- (b + 1)]) > TRIM_THRESHOLD * w * 255:
                flag = True
            else:
                if flag:
                    break
        if b == TRIM_ROWS_TO_SCAN - 1:
            b = 0

        # Left
        l = 0
        flag = False
        for l in range(TRIM_ROWS_TO_SCAN):
            if sum(ans_img_thr[:, l]) > TRIM_THRESHOLD * h * 255:
                flag = True
            else:
                if flag:
                    break
        if l == TRIM_ROWS_TO_SCAN - 1:
            l = 0

        # Right
        r = 0
        flag = False
        for r in range(TRIM_ROWS_TO_SCAN):
            if sum(ans_img_thr[:, - (r + 1)]) > TRIM_THRESHOLD * h * 255:
                flag = True
            else:
                if flag:
                    break
        if r == TRIM_ROWS_TO_SCAN - 1:
            r = 0

        return ans_img_raw[t:h - b, l:w - r], ans_img_thr[t:h - b, l:w - r]

    def read_all_answers(self):
        """
        Read the selection of each answer block
        Only one selections allowed, high reliability
        Read from the raw image because Gaussian Adaptive Threshold treats any large 
		block of content, either dark or bright, as background. Therefore, only the edges 
		of the circles is detected. We need the inside content of the circle for accuracy.
        """
        self.marked_ans = []
        fun_sort_by_val = lambda entry: entry[1]
        for i in range(self.num_questions):
            img = self.answer_imgs[i] \
                    [TOP_BOTTOM_MARGIN:-TOP_BOTTOM_MARGIN,
                     LEFT_RIGHT_MARGIN:-LEFT_RIGHT_MARGIN]
            h, w = img.shape[:2]
            #print h, w
            #plt.subplot(10, 6, i + 1)
            #plt.imshow(img)
            agg_vals = []
            for i in range(5):
                region = img[:, int(i * (w / 5.)) : int((i + 1) * (w / 5.))]
                agg_vals.append(('ABCDE'[i], np.sum(region)))
            #print 'aggval', agg_vals
            agg_vals = sorted(agg_vals, key=fun_sort_by_val)
            selection = agg_vals[0][0]
            self.marked_ans.append(selection)
        #print 'marked answer', self.marked_ans

    def read_datamatrix(self):
        datamatrix = self.raw_img[QR_ROW_MIN:QR_ROW_MAX, QR_COL_MIN:QR_COL_MAX]
        plt.imshow(datamatrix)
        plt.show()
        try:
            content = pylibdmtx.decode(datamatrix, timeout=QR_TIMEOUT)[0][0]
            print 'qr content', content
            self.test_id = content[:5]
            self.paper_id = content[5:8]
        except Exception as e:
            print '[WARNING] Cannot decode datamatrix:', e
            self.test_id = None
            self.paper_id = None


rp = RawPhoto('test_four0.jpg')
res = rp.read_content(4)
print res
plt.show()
