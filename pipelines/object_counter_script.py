import json
data = {}
LABELS = []
COUNTER = {}


def send():
    b = Buffer(80)
    b.setData(json.dumps(COUNTER).encode('utf-8'))
    node.io['out'].send(b)


def tracklet_removed(tracklet, coords2):
    coords1 = tracklet['coords']
    deltaX = coords2[0] - coords1[0]
    deltaY = coords2[1] - coords1[1]

    if abs(deltaY) > THRESH_DIST_DELTA and 0 < deltaY:
        labelStr = LABELS[t.label]
        COUNTER[labelStr] += 1
        send()
        node.warn(f"Cargo detected")
        node.warn("{}:{}".format(labelStr, COUNTER[labelStr]))


def get_centroid(roi):
    x1 = roi.topLeft().x
    y1 = roi.topLeft().y
    x2 = roi.bottomRight().x
    y2 = roi.bottomRight().y
    return ((x2-x1)/2+x1, (y2-y1)/2+y1)


# Send dictionary initially (all counters 0)
send()

while True:
    tracklets = node.io['tracklets'].get()
    for t in tracklets.tracklets:
        # If new tracklet, save its centroid
        if t.status == Tracklet.TrackingStatus.NEW:
            data[str(t.id)] = {} # Reset
            data[str(t.id)]['coords'] = get_centroid(t.roi)
        elif t.status == Tracklet.TrackingStatus.TRACKED:
            data[str(t.id)]['lostCnt'] = 0
        elif t.status == Tracklet.TrackingStatus.LOST:
            data[str(t.id)]['lostCnt'] += 1
            # If tracklet has been "LOST" for more than 10 frames, remove it
            if 10 < data[str(t.id)]['lostCnt'] and "lost" not in data[str(t.id)]:
                #node.warn(f"Tracklet {t.id} lost: {data[str(t.id)]['lostCnt']}")
                tracklet_removed(data[str(t.id)], get_centroid(t.roi))
                data[str(t.id)]["lost"] = True
        elif (t.status == Tracklet.TrackingStatus.REMOVED) and "lost" not in data[str(t.id)]:
            tracklet_removed(data[str(t.id)], get_centroid(t.roi))
            #node.warn(f"Tracklet {t.id} removed")