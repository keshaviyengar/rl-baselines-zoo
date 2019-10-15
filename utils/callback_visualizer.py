import rospy
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField


import struct
import numpy as np

import os
from functools import reduce

MM_TO_M = 1000


def size_type_to_datatype(size, type):
    """
    Given a .pcd size/type pair, return a sensor_msgs/PointField datatype
    """
    if type == "F":
        if size == 4:
            return 7
        if size == 8:
            return 8
    if type == "I":
        if size == 1:
            return 1
        if size == 2:
            return 3
        if size == 4:
            return 5
    if type == "U":
        if size == 1:
            return 2
        if size == 2:
            return 4
        if size == 4:
            return 6
    raise Exception("Unknown size/type pair in .pcd")


def datatype_to_size_type(datatype):
    """
    Takes a UINT8 datatype field from a PointFields and returns the size in
    bytes and a char for its type ('I': int, 'U': unsigned, 'F': float)
    """
    # might be nicer to look it up in message definition but quicker just to
    # do this.
    if datatype in [2, 4, 6]:
        t = 'U'
    elif datatype in [1, 3, 5]:
        t = 'I'
    elif datatype in [7, 8]:
        t = 'F'
    else:
        raise Exception("Unknown datatype in PointField")

    if datatype < 3:
        s = 1
    elif datatype < 5:
        s = 2
    elif datatype < 8:
        s = 4
    elif datatype < 9:
        s = 8
    else:
        raise Exception("Unknown datatype in PointField")

    return s, t


def read_pcd(filename, cloud_header=None):
    if not os.path.isfile(filename):
        raise Exception("[read_pcd] File does not exist.")
    string_array = lambda x: x.split()
    float_array = lambda x: [float(j) for j in x.split()]
    int_array = lambda x: [int(j) for j in x.split()]
    word = lambda x: x.strip()
    headers = [("VERSION", float),
               ("FIELDS", string_array),
               ("SIZE", int_array),
               ("TYPE", string_array),
               ("COUNT", int_array),
               ("WIDTH", int),
               ("HEIGHT", int),
               ("VIEWPOINT", float_array),
               ("POINTS", int),
               ("DATA", word)]
    header = {}
    with open(filename, "rb") as pcdfile:
        while len(headers) > 0:
            line = pcdfile.readline().decode('utf8')
            if line == "":
                raise Exception("[read_pcd] EOF reached while looking for headers.")
            f, v = line.split(" ", 1)
            if f.startswith("#"):
                continue
            if f not in list(zip(*headers))[0]:
                raise Exception("[read_pcd] Field '{}' not known or duplicate.".format(f))
            func = headers[list(zip(*headers))[0].index(f)][1]
            header[f] = func(v)
            headers.remove((f, func))
        data = pcdfile.read()
    # Check the number of points
    if header["VERSION"] != 0.7:
        raise Exception("[read_pcd] only PCD version 0.7 is understood.")
    if header["DATA"] != "binary":
        raise Exception("[read_pcd] Only binary .pcd files are readable.")
    if header["WIDTH"] * header["HEIGHT"] != header["POINTS"]:
        raise Exception("[read_pcd] POINTS count does not equal WIDTH*HEIGHT")

    cloud = PointCloud2()
    cloud.point_step = sum([size * count
                            for size, count in zip(header["SIZE"], header["COUNT"])])
    cloud.height = header["HEIGHT"]
    cloud.width = header["WIDTH"]
    cloud.row_step = cloud.width * cloud.point_step
    cloud.is_bigendian = False
    if cloud.row_step * cloud.height > len(data):
        raise Exception("[read_pcd] Data size mismatch.")
    offset = 0
    for field, size, type, count in zip(header["FIELDS"],
                                        header["SIZE"],
                                        header["TYPE"],
                                        header["COUNT"]):

        if field != "_":
            pass
        pf = PointField()
        pf.count = count
        pf.offset = offset
        pf.name = field
        pf.datatype = size_type_to_datatype(size, type)
        cloud.fields.append(pf)
        offset += size * count

    cloud.data = data[0:cloud.row_step * cloud.height]
    if cloud_header is not None:
        cloud.header = header
    else:
        cloud.header.frame_id = "/pcd_cloud"
    return cloud


def write_pcd(filename, pointcloud, overwrite=False, viewpoint=None,
              mode="binary"):
    """
    Writes a sensor_msgs::PointCloud2 to a .pcd file.
    :param filename - the pcd file to write
    :param pointcloud - sensor_msgs::PointCloud2 to write to a file
    :param overwrite - if True, allow overwriting existing files
    :param viewpoint - the camera viewpoint, (x,y,z,qw,qx,qy,qz)
    :param mode - the writing mode: 'ascii' for human readable, 'binary' for
                  a straight dump of the binary data, 'binary_stripped'
                  to strip out data padding before writing (saves space but it slow)
    """
    assert isinstance(pointcloud, PointCloud2)
    if mode not in ['ascii', 'binary', 'binary_stripped']:
        raise Exception("Mode must be 'binary' or 'ascii'")
    if not overwrite and os.path.isfile(filename):
        raise Exception("File exists.")
    try:
        with open(filename, "wb") as f:
            f.write("VERSION .7\n".encode('utf8'))
            _size = {}
            _type = {}
            _count = {}
            _offsets = {}
            _fields = []
            _size['_'] = 1
            _count['_'] = 1
            _type['_'] = 'U'
            offset = 0
            for field in pointcloud.fields:
                if field.offset != offset:
                    # some padding
                    _fields.extend(['_']*(field.offset - offset))
                isinstance(field, PointField)
                _size[field.name], _type[field.name] = datatype_to_size_type(field.datatype)
                _count[field.name] = field.count
                _offsets[field.name] = field.offset
                _fields.append(field.name)
                offset = field.offset + _size[field.name] * _count[field.name]
            if pointcloud.point_step != offset:
                _fields.extend(['_']*(pointcloud.point_step - offset))

            if mode != 'binary':
                while True:
                    try:
                        _fields.remove('_')
                    except:
                        break

            _fields_str = reduce(lambda a, b: a + ' ' + b,
                                  map(lambda x: "{%s}" % x,
                                      _fields))

            f.write("FIELDS ".encode('utf8'))
            f.write(reduce(lambda a, b: a + ' ' + b,
                           _fields).encode('utf8'))
            f.write("\n".encode('utf8'))
            f.write("SIZE ".encode('utf8'))
            f.write(_fields_str.format(**_size).encode('utf8'))
            f.write("\n".encode('utf8'))
            f.write("TYPE ".encode('utf8'))
            f.write(_fields_str.format(**_type).encode('utf8'))
            f.write("\n".encode('utf8'))
            f.write("COUNT ".encode('utf8'))
            f.write(_fields_str.format(**_count).encode('utf8'))

            f.write("\n".encode('utf8'))
            f.write(("WIDTH %s" % pointcloud.width).encode('utf8'))
            f.write("\n".encode('utf8'))
            f.write(("HEIGHT %s" % pointcloud.height).encode('utf8'))
            f.write("\n".encode('utf8'))

            if viewpoint is None:
                f.write("VIEWPOINT 0 0 0 1 0 0 0\n".encode('utf8'))
            else:
                try:
                    assert len(viewpoint) == 7
                except AssertionError:
                    raise Exception("viewpoint argument must be  tuple "
                                    "(x y z qx qy qz qw)")
                f.write("VIEWPOINT {} {} {} {} {} {} {}\n".format(*viewpoint).encode('utf8'))

            f.write("POINTS %d\n".encode('utf8') % (pointcloud.width * pointcloud.height))
            if mode == "binary":
                #TODO: check for row padding.
                f.write("DATA binary\n".encode('utf8'))
                f.write(bytearray(pointcloud.data))
            elif mode == "binary_stripped":
                f.write("DATA binary\n".encode('utf8'))
                if pointcloud.point_step == sum([v[0]*v[1] for v in zip(_size.values(),
                                                                        _count.values())]): #danger, assumes ordering
                    # ok to just blast it all out; TODO: this assumes row step has no padding
                    f.write(bytearray(pointcloud.data))
                else:
                    # strip all the data padding
                    _field_step = {}
                    for field in _fields:
                        _field_step[field] = _size[field] * _count[field]
                    out = bytearray(sum(_field_step.values())*pointcloud.width*pointcloud.height)
                    b = 0
                    for v in range(pointcloud.height):
                        offset = pointcloud.row_step * v
                        for u in range(pointcloud.width):
                            for field in _fields:
                                out[b:b+_field_step[field]] = pointcloud.data[offset+_offsets[field]:
                                                                              offset+_offsets[field]+_field_step[field]]
                                b += _field_step[field]
                            offset += pointcloud.point_step
                    f.write(out)
            else:
                f.write("DATA ascii\n".encode('utf8'))
                for p in pcl2.read_points(pointcloud,  _fields):
                    for i, field in enumerate(_fields):
                        f.write("%f ".encode('utf8') % p[i])
                    f.write("\n".encode('utf8'))

    except IOError as e:
        raise Exception("Can't write to %s: %s" % (filename, e.message))


# Goal of this class to visualize concentric tube robot training in rviz.
# Publish joint states during training and / or evaluation.
# Publish training point cloud (reached cartesian point and associated error / q-value)
# Publish a voxel grid of training locations
class CallbackVisualizer(object):
    def __init__(self):
        self._locals = None
        self._globals = None
        self.pcl_points = np.array([])
        self.ag_points = np.array([])
        self.errors = np.array([])
        self.q_values = np.array([])
        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                       PointField('y', 4, PointField.FLOAT32, 1),
                       PointField('z', 8, PointField.FLOAT32, 1),
                       PointField('rgba', 12, PointField.UINT32, 1)]

        self.q_value_pcl = None
        self.error_pcl = None

        self.q_value_pcl_pub = rospy.Publisher('ctm/q_value_pcl', PointCloud2, queue_size=10)
        self.error_pcl_pub = rospy.Publisher('ctm/error_pcl', PointCloud2, queue_size=10)
        rospy.init_node("pcl_visualizer", anonymous=True)

    def callback(self, _locals, _globals):
        self._locals = _locals
        self._globals = _globals
        observation = _locals['self'].env.convert_obs_to_dict(_locals['new_obs'])
        ag = observation['achieved_goal'] * MM_TO_M
        self.ag_points = np.append(self.ag_points, ag)
        error = _locals['info']['error']
        self.errors = np.append(self.errors, error)
        q_val = _locals['q_value']
        self.q_values = np.append(self.q_values, q_val)

        if _locals['total_steps'] % 1000 == 0:
            self.save_point_clouds()
            self.publish_point_clouds()

    def save_point_clouds(self):
        array_to_rgba_array = np.vectorize(self._value_to_int32_rgba)

        normalized_rgba = np.squeeze(array_to_rgba_array(self.q_values, np.min(self.q_values), np.max(self.q_values)))
        ag_points = self.ag_points.reshape(-1, 3).transpose()
        ag_pcl_points = np.ndarray((4, ag_points.shape[1]), dtype=object)
        ag_pcl_points[0:3, 0:] = ag_points
        ag_pcl_points[3, :] = normalized_rgba

        normalized_error = np.squeeze(array_to_rgba_array(self.errors, np.min(self.errors), np.max(self.errors)))
        error_pcl_points = np.ndarray((4, ag_points.shape[1]), dtype=object)
        error_pcl_points[0:3, 0:] = ag_points
        error_pcl_points[3, :] = normalized_error

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"
        # Normalize the rgb values before publishing.
        self.q_value_pcl = pcl2.create_cloud(header, self.fields, ag_pcl_points.transpose().tolist())
        self.error_pcl = pcl2.create_cloud(header, self.fields, error_pcl_points.transpose().tolist())

        write_pcd('error.pcd', self.q_value_pcl, overwrite=True)
        write_pcd('ag.pcd', self.error_pcl, overwrite=True)

    def publish_point_clouds(self):
        self.q_value_pcl_pub.publish(self.q_value_pcl)
        self.error_pcl_pub.publish(self.error_pcl)

        # How to read and publish read point cloud
        # test_cloud = read_pcd('test.pcd')
        # test_cloud.header.stamp = rospy.Time.now()
        # test_cloud.header.frame_id = "world"
        # self.error_pcl_pub.publish(test_cloud)

    @staticmethod
    def _value_to_int32_rgba(value, minimum, maximum):
        minimum, maximum = float(minimum), float(maximum)
        ratio = 2 * (value - minimum) / (maximum - minimum)
        b = int(max(0, 255 * (1 - ratio)))
        r = int(max(0, 255 * (ratio - 1)))
        g = 255 - b - r
        a = 255
        rgba = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        return rgba
