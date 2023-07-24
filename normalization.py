import numpy as np

def stroke_length(stroke):
    L = 0
    L_n = [0]
    for i in range(len(stroke)-1):
        L = L + np.sqrt((stroke[i+1][0] - stroke[i][0])**2 + (stroke[i+1][1] - stroke[i][1])**2)
        L_n.append(L)
    return L_n


def Speed_norm_stroke(stroke, norm_nb, alpha):
    new_stroke = []

    # calculate stroke length
    L_n = stroke_length(stroke)
    
    m = int(L_n[-1]/alpha)
    new_stroke.append(stroke[0])
    j = 1
    for p in range(m - 2):
        while L_n[j] < alpha*p:
            j = j + 1
        c = (alpha*p - L_n[j-1])/(L_n[j] - L_n[j-1])
        new_point = (stroke[j - 1][0] + c*(stroke[j][0] - stroke[j-1][0]), stroke[j-1][1] + c*(stroke[j][1] - stroke[j-1][1]))
        new_stroke.append(new_point)
    new_stroke.append(stroke[-1])
        
    return np.array(new_stroke)

def normalization(strokes, norm_nb, norm_type = 'stroke_keep_shape', speed_norm = False):
    stroke_level = np.zeros((len(strokes), norm_nb, 2), dtype=np.float32)
    # eq_level = np.zeros((len(strokes), norm_nb, 2), dtype=np.float32)
    # calculate stroke length
    max_len = max(stroke_length(stroke)[-1] for stroke in strokes)
    alpha = max_len/norm_nb
    new_equation = []
    if speed_norm == True:
        for i in range(len(strokes)):
            n = np.zeros((norm_nb, 2))
            new_stroke = Speed_norm_stroke(strokes[i], alpha)
            if norm_type == 'stroke_keep_shape':
                new_stroke = Spatial_norm_stroke_keep_shape(new_stroke)
            elif norm_type == 'stroke_tension':
                new_stroke = Spatial_norm_stroke_tension(new_stroke)
            # print(len(new_stroke))
            n[:len(new_stroke), :] = new_stroke
            stroke_level[i, :, :] = n
            new_equation.append(new_stroke)
        eq_level = np.array(Spatial_norm_eq_keep_shape(new_equation, norm_nb))
    else:
        stroke_level, new_list = Simple_norm_equation(strokes, norm_nb)
        eq_level = np.array(Spatial_norm_eq_keep_shape(new_list, norm_nb))
    return stroke_level, eq_level

def eq_normalization(strokes, norm_nb, norm_type = 'stroke_keep_shape', speed_norm = False):
    new_equation = np.zeros((len(strokes), norm_nb, 2), dtype=np.float32)


def Simple_norm_equation(strokes, norm_nb):
    new_strokes = np.zeros((len(strokes), norm_nb, 2), dtype=np.float32)
    new_list = []
    for i in range(len(strokes)):
        # s = Spatial_norm_stroke_tension(strokes[i])
        s = strokes[i]
        if len(s) < norm_nb:
            new_stroke = np.zeros((norm_nb, 2))
            new_stroke[:len(s), :] = s
        else:
            new_stroke = multi_sampling(s, norm_nb)

        if np.isnan(new_stroke).any() == True:
            print('norm error')
        else:
            new_strokes[i, :, :] = new_stroke
            new_list.append(new_stroke)

    return new_strokes, new_list

def No_speed_norm_stroke(stroke, norm_nb, alpha):
    if len(stroke) < norm_nb:
        return np.array(stroke)
    else:
        return multi_sampling(stroke, norm_nb)

def multi_sampling(stroke, norm_nb):
    new_stroke = np.zeros((len(stroke)*norm_nb, 2))
    for i in range(len(stroke)):
        new_stroke[i*norm_nb:(i+1)*norm_nb, :] = stroke[i]
        # new_stroke[i*norm_nb:(i+1)*norm_nb, :] = stroke[:,i].expand(norm_nb, 2).reshape(2,-1)
    points = np.zeros((norm_nb, 2))
    for i in range(norm_nb):
        points[i,:] = np.mean(new_stroke[i*len(stroke): (i+1)*len(stroke),:])
    return points

# normalize stroke y to [0, 1], keep the shape
def stroke_keep_shape(stroke):
    s = stroke.reshape([-1, 2])
    if np.min(s[:, 1]) != np.max(s[:, 1]):
        s[:, 1] = (s[:, 1] - np.min(s[:, 1]))/(np.max(s[:, 1]) - np.min(s[:, 1]))
        if np.min(s[:, 0]) != np.max(s[:, 0]):
            s[:, 0] = (s[:, 0] - np.min(s[:, 0]))/(np.max(s[:, 1]) - np.min(s[:, 1]))
        else:
            s[:, 0] = 0.5
    else:
        s[:, 1] = 0.5
        if np.min(s[:, 0]) != np.max(s[:, 0]):
            s[:, 0] = (s[:, 0] - np.min(s[:, 0]))/(np.max(s[:, 0]) - np.min(s[:, 0]))
        else:
            s[:, 0] = 0.5
    return s

def stroke_tension(stroke):
    s = stroke.reshape([-1, 2])
    if np.min(s[:, 0]) != np.max(s[:, 0]):
        s[:, 0] = (s[:, 0] - np.min(s[:, 0]))/(np.max(s[:, 0]) - np.min(s[:, 0]))
    else:
        s[:, 0] = 0.5
    if np.min(s[:, 1]) != np.max(s[:, 1]):
        s[:, 1] = (s[:, 1] - np.min(s[:, 1]))/(np.max(s[:, 1]) - np.min(s[:, 1]))
    else:
        s[:, 1] = 0.5
    return s

def eq_keep_shape(strokes, norm_nb):
    max_x, max_y, min_x, min_y = get_max_min(strokes)
    new_strokes = np.zeros((len(strokes), norm_nb, 2), dtype=np.float32)
    # new_strokes = []
    for i in range(len(strokes)):
        new_stroke = []
        for point in strokes[i]:
            new_point = ((point[0] - min_x)/(max_y - min_y), (point[1] - min_y)/(max_y - min_y))
            new_stroke.append(new_point)
        n = np.zeros((norm_nb, 2))
        n[:len(new_stroke),:] = new_stroke
        new_strokes[i, :, :] = n
        # new_strokes.append(new_stroke)
    return new_strokes

# def padding_fixed_length(strokes, norm_nb, type = 'stroke_keep_shape', speed_norm = False):
    if speed_norm == True:
        eq = Speed_norm_equation(strokes, norm_nb)
    else:
        eq = Simple_norm_equation(strokes, norm_nb)
    new_strokes = np.zeros((len(eq), norm_nb, 2), dtype=np.float32)
    if type == 'eq_keep_shape':
        new_list = Spatial_norm_eq_keep_shape(eq)
        for i in range(len(new_list)):
            new_strokes[i, :, :] = new_list[i]
    else:
        for i in range(len(eq)):
            if type == 'stroke_keep_shape':
                s = Spatial_norm_stroke_keep_shape(eq[i])
            else:
                s = Spatial_norm_stroke_tension(eq[i])
            new_stroke = np.zeros((norm_nb, 2))
            new_stroke[:s.shape[0], :] = s

            if np.isnan(new_stroke).any() == True:
                print('norm error')
            else:
                new_strokes[i, :, :] = new_stroke

    return new_strokes

def get_max_min(strokes):
    max_x = 0
    max_y = 0
    min_x = 0
    min_y = 0
    for stroke in strokes:
        for point in stroke:
            if point[0] > max_x:
                max_x = point[0]
            if point[0] < min_x:
                min_x = point[0]
            if point[1] > max_y:
                max_y = point[1]
            if point[1] < min_y:
                min_y = point[1]
    return max_x, max_y, min_x, min_y

def norm_eq(strokes):
    max_x, max_y, min_x, min_y = get_max_min(strokes)
    new_strokes = []
    for stroke in strokes:
        new_stroke = []
        for point in stroke:
            new_point = ((point[0] - min_x)/(max_y - min_y), (point[1] - min_y)/(max_y - min_y))
            new_stroke.append(new_point)
        new_strokes.append(new_stroke)
    return new_strokes