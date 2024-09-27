import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

# # ----------------- Main ----------------- # #


def metric(paths:list, preds:list, labels:list, needs=[]):
    """
    :param paths:
    :param preds:
    :param labels:
    :param needs: list, including 'map', 'ap', 'miou'
    :param is_soft:
    :return:
    """
    moviePL = collectMovie(paths, preds, labels)
    met = {}
    if 'map'in needs:
        map = callmAP(moviePL)
        met.update({'mAP': map})
    if 'ap' in needs:
        ap = callAP(moviePL)
        met.update({'AP': ap})
    if 'miou' in needs:
        miou = 0.0
        for movie in moviePL.keys():
            pd, lb = moviePL[movie]
            pd = result2dict(pd)
            lb = result2dict(lb)
            iou = callMIOU(lb, pd)
            miou += iou
        miou = miou/len(moviePL.keys())
        met.update({'mIoU': miou})
    if 'f1' in needs:
        f1 = callF1(moviePL)
        met.update({'F1': f1})
    return met, moviePL


def metric_scrl(paths:list, preds:list, labels:list, needs=[]):
    """
    :param paths:
    :param preds:
    :param labels:
    :param needs: list, including 'map', 'ap', 'miou'
    :param is_soft:
    :return:
    """
    moviePL = collectMovie(paths, preds, labels)
    met = {}
    if 'map'in needs:
        map = callmAP(moviePL)
        met.update({'mAP': map})
    if 'ap' in needs:
        ap = callAP(moviePL)
        met.update({'AP': ap})
    if 'miou' in needs:
        miou = 0.0
        for movie in moviePL.keys():
            pd, lb = moviePL[movie]
            pd = result2dict(pd)
            lb = result2dict(lb)
            iou = callMIOU(lb, pd)
            miou += iou
        miou = miou/len(moviePL.keys())
        met.update({'mIoU': miou})
    if 'f1' in needs:
        f1 = callF1(moviePL)
        met.update({'F1': f1})
    return met


# # ----------------- IoU ----------------- # #


def result2dict(result):
    sceneDict = {}
    sceneDict.update({0: [0]})
    sid = 0
    spt, ept = 0, 0
    for pt in range(result.shape[0]):
        if result[pt] > 0.5:
            # allocate
            ept = pt
            sceneDict[sid].append(ept)
            # update
            spt = pt
            sid += 1
            sceneDict.update({sid: [spt]})
        if pt == result.shape[0] - 1:
            sceneDict[sid].append(pt)

    return sceneDict


def callMIOU(groundSceneDict, predSceneDict):
    """
    Calculating Mean IoU for a movie
    :param groundSceneDict: dict, {shot_index: (start, end)}
    :param predSceneDict: dict, {shot_index: (start, end)}
    :return:
    """
    ious_g = callIOU(groundSceneDict, predSceneDict)
    ious_p = callIOU(predSceneDict, groundSceneDict)
    miou_g = [iou for _, iou, _ in ious_g]
    miou_p = [iou for _, iou, _ in ious_p]
    miou = 0.5*np.mean(miou_g) + 0.5*np.mean(miou_p)
    return miou


def callIOU(groundSceneDict, predSceneDict):
    """
    A movie iou for scenes
    :param groundSceneDict: dict, {shot_index: (start, end)}
    :param predSceneDict: dict, {shot_index: (start, end)}
    :return:
        iou: list, each element is (sceneid, max_rad, max_pred_id), one movie iou for each scene
    """
    iou = []
    for sceneid in groundSceneDict.keys():
        ratios = []
        gtScene = groundSceneDict[sceneid]
        for pred_id, predScene in predSceneDict.items():
            rat = _getRatio(gtScene, predScene)
            ratios.append([rat, pred_id])
        ratios = np.array(ratios)
        max_rat = np.max(ratios[:, 0])
        max_pred_id = np.argmax(ratios[:, 0])
        max_pred_id = ratios[max_pred_id, 1]
        iou.append((sceneid, max_rat, max_pred_id))

    return iou


def _getRatio(interval_1, interval_2):
    """
    Calculating IoU for interval_1 and interval_2
    :param interval_1: set, (start, end)
    :param interval_2: set, (start, end)
    :return:
        ratio, scale
    """
    interaction = _getInteraction(interval_1, interval_2)
    if interaction == 0:
        return 0
    return interaction/_getUnion(interval_1, interval_2)


def _getInteraction(interval_1, interval_2):
    """
    Calculating interaction of interval_1 and interval_2
    :param interval_1:
    :param interval_2:
    :return:
    """
    s1 = interval_1[0]
    e1 = interval_1[1]
    s2 = interval_2[0]
    e2 = interval_2[1]
    start = max(s1, s2)
    end = min(e1, e2)

    if start <= end:
        if start == end:
            return 1
        else:
            return (end - start)
    return 0


def _getUnion(interval_1, interval_2):
    """
    Calculating Union of interval_1 and interval_2
    :param interval_1:
    :param interval_2:
    :return:
    """
    s1 = interval_1[0]
    e1 = interval_1[1]
    s2 = interval_2[0]
    e2 = interval_2[1]
    start = min(s1, s2)
    end = max(e1, e2)

    if start == end:
        return 1
    return (end - start)


# # ----------------- mAP ----------------- # #


def callmAP(moviePL:dict):
    map = 0
    n_movie = len(moviePL.keys())
    for movie in moviePL.keys():
        pred, label = moviePL[movie]
        # pred = (pred>=0.5)*1
        ap = average_precision_score(label, pred)
        map += ap
    return map/n_movie


def callAP(moviePL:dict):
    preds = None
    labels = None
    for movie in moviePL.keys():
        pd, lb = moviePL[movie]
        if preds is None:
            preds = pd
            labels = lb
        else:
            preds = np.concatenate((preds, pd), axis=0)
            labels = np.concatenate((labels, lb), axis=0)
    ap = average_precision_score(labels, preds)
    return ap


def callF1(moviePL:dict):
    preds = None
    labels = None
    for movie in moviePL.keys():
        pd, lb = moviePL[movie]
        pd = (pd >= 0.5) * 1
        if preds is None:
            preds = pd
            labels = lb
        else:
            preds = np.concatenate((preds, pd), axis=0)
            labels = np.concatenate((labels, lb), axis=0)
    f1 = f1_score(labels, preds)
    return f1


# # ----------------- Precess Soft Label ----------------- # #


def postprocess_soft(preds):
    """
    :param preds: (n_shot)
    :return:
    """
    n_shot = preds.shape[0]
    pred_hard = np.zeros(n_shot)
    index = preds > 0.5
    pred_hard[index] = 1

    return pred_hard


# # ----------------- Prepare Work ----------------- # #


def collectMovie(paths:list, preds:list, labels:list):
    """

    :param paths: each element of the list is a tuple
    :param preds: each element of the list is ndarray
    :param labels: each element of the list is tensor
    :return:
        MoviePL: dict, each moviePL['xxx'] is a list,
                where the list contains pred_nd, label_nd
        pred_nd: (n_shot)
        label_nd: same above
    """
    pathlist = []
    predlist = []
    labelist = []
    for pth, prd, lab in zip(paths, preds, labels):
        pthlist = [it for it in pth]
        pathlist.append(pthlist)
        predlist.append(prd)
        labelist.append(lab)

    moviePL = {}
    for pth_batch, prd_batch, lab_batch in zip(pathlist, predlist, labelist):
        for pth, prd, lab in zip(pth_batch, prd_batch, lab_batch):
            movie_id, shot_id = parse_path(pth)
            if movie_id not in moviePL.keys():
                moviePL.update({movie_id: {shot_id: [prd, lab]}})
            else:
                moviePL[movie_id].update({shot_id:[prd, lab]})

    for movie in moviePL.keys():
        PL = moviePL[movie]
        n_shot = len(PL.keys()) - 10
        pred_nd = np.zeros(n_shot)
        label_nd = np.zeros(n_shot)

        for i in PL.keys():
            if i >= 10:
                j = i-10
                pd, lb = PL[j]
                pred_nd[j] = pd
                label_nd[j] = lb

        moviePL[movie] = [pred_nd, label_nd]

    return moviePL


def parse_path(path:str):
    path = path.split('/')[-1]
    movie_id, remain = path.split('shot')
    shot_id = remain[:-4]

    return movie_id[:-1], int(shot_id)

if __name__=='__main__':
    pass

