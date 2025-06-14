from dataclasses import dataclass
import math


@dataclass(unsafe_hash=True, frozen=True)
class Vector_2D:
    x: float
    y: float

    # makes using these fields more memory and runtime efficient
    __slots__ = ['x', 'y']

    def __add__(self, other: "Vector_2D") -> "Vector_2D":
        return Vector_2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector_2D") -> "Vector_2D":
        return Vector_2D(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float) -> "Vector_2D":
        return Vector_2D(self.x * other, self.y * other)

    def __rmul__(self, other: float) -> "Vector_2D":
        return Vector_2D(self.x * other, self.y * other)

    def __truediv__(self, other: float) -> "Vector_2D":
        return Vector_2D(self.x / other, self.y / other)

    def __neg__(self) -> "Vector_2D":
        return Vector_2D(-self.x, -self.y)

    def round(self) -> "Vector_2D":
        """
        return the same vector, with its coordinates rounded to the nearest integer
        """
        return Vector_2D(round(self.x), round(self.y))

    def dot(self, other: "Vector_2D") -> float:
        return self.x * other.x + self.y * other.y

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def distance(self, other: "Vector_2D") -> float:
        return (other - self).length()

    def signed_area(self, other: "Vector_2D") -> float:
        """
        find the signed area of the parallelogram formed by the vectors.
        in 3D this would be the length of the cross product.
        """
        return self.x * other.y - self.y * other.x

    def angle(self, other: "Vector_2D") -> float:
        """
        find the angle from this vector to the other, in radians
        """
        return math.atan2(self.signed_area(other), self.dot(other))

    def rotate(self, angle: float) -> "Vector_2D":
        """
        rotate the vector by the angle, given in radians
        """
        cos = math.cos(angle)
        sin = math.sin(angle)
        return Vector_2D(self.x * cos - self.y * sin,
                    self.x * sin + self.y * cos)

    def project(self, other: "Vector_2D") -> "Vector_2D":
        """
        returns the projection of the other vector onto this one
        """
        # multiplication by the othogonal projection matrix (v*v^T)/(v^T*v)
        x = self.x * self.x * other.x + self.x * self.y * other.y
        y = self.x * self.y * other.x + self.y * self.y * other.y
        squared_length = self.dot(self)
        if squared_length < 0.0001:
            # this vector is so small the projection is onto a point,
            # which is just the point itself
            return self
        return Vector_2D(x, y) / self.dot(self)

    def normalize(self) -> "Vector_2D":
        length = self.length()
        if length < 0.0001:
            # avoid dividing by zero
            return Vector_2D(0, 0)
        return self / length


@dataclass(unsafe_hash=True, frozen=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float

    # makes using these fields more memory and runtime efficient
    __slots__ = ['x', 'y', 'z', 'w']

    @staticmethod
    def from_euler_angles(roll, pitch, yaw):
        t0 = math.cos(yaw * 0.5)
        t1 = math.sin(yaw * 0.5)
        t2 = math.cos(roll * 0.5)
        t3 = math.sin(roll * 0.5)
        t4 = math.cos(pitch * 0.5)
        t5 = math.sin(pitch * 0.5)

        w = t0 * t2 * t4 + t1 * t3 * t5
        x = t0 * t3 * t4 - t1 * t2 * t5
        y = t0 * t2 * t5 + t1 * t3 * t4
        z = t1 * t2 * t4 - t0 * t3 * t5
        return Quaternion(x, y, z, w)

    def __mul__(self, other: "Quaternion"):
        t, x, y, z = self.w, self.x, self.y, self.z
        a, b, c, d = other.w, other.x, other.y, other.z
        return Quaternion(w=a * t - b * x - c * y - d * z,
                          x=b * t + a * x + d * y - c * z,
                          y=c * t + a * y + b * z - d * x,
                          z=d * t + z * a + c * x - b * y)

    def conjugate(self):
        return Quaternion(-self.x, -self.y, -self.z, self.w)


def getFoVCoverage(center: Vector_2D, radius: float) :
    origin = Vector_2D(0, 0)
    dist = origin.distance(center)
    if dist <= radius:
        return None
    return math.atan2(radius, math.sqrt(dist**2 - radius**2))


def checkoverlapCircle(a: Vector_2D, b: Vector_2D, o: Vector_2D, radius: float):
    if o.distance(a) < radius or o.distance(b) < radius:
        return True

    # find the projection of o onto the line passing through a and b
    line = b - a
    p = line.project(o - a) + a
    if o.distance(p) > radius:
        # if the projection is outside the circle,
        # all other points would be further away from the center,
        # and therefore not inside the circle
        return False
    # since a and b are outside the circle,
    # the segment overlaps iff the segment (a,b) intersects with the ray from o to p
    #
    # if the points are on opposite sides, the angle between them would be pi
    # and 0 if they are on the same side
    return abs((b - p).angle(a - p)) > math.pi / 2