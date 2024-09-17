import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from dataclasses import dataclass, field


@dataclass
class DieGeometry:
    enscribed_perimeter_ratio:float = field(default=1.0)
    circumscribed_face_ratio:float = field(default=1.0)
    enscribed_face_ratio:float = field(default=1.0)
    #pixel_comparison_ratio:float
    top_face_ratio:float = field(default=1.0)
    adjacent_face_ratio:float = field(default=1.0) #includes the centroid of all profiles on an adjacent face
    perimeter_edges:int = field(default=1)
    face_edges:int = field(default=1)
    adjacent_faces:int = field(default=0)
    adjacent_face_angle:float = field(default=0) #degrees

    @staticmethod
    def get_common_die_geometry(rank:int):
        g = DieGeometry()
        match rank:
            case 1:
                g.enscribed_perimeter_ratio = 1.0
                g.circumscribed_face_ratio = 1.0
                g.enscribed_face_ratio = 1.0
                g.top_face_ratio = g.enscribed_face_ratio
                g.adjacent_face_ratio = g.enscribed_face_ratio
                g.perimeter_edges = 1
                g.face_edges = 1
                g.adjacent_faces = 0
            case 4:
                # tetrahedron:
                # g.enscribed_perimeter_ratio = np.cos(np.pi/3)
                # g.circumscribed_face_ratio = 1
                # g.enscribed_face_ratio = np.cos(np.pi/3)
                g.top_face_ratio = g.enscribed_face_ratio
                # g.pixel_comparison_ratio = g.enscribed_face_ratio
                # g.perimeter_edges = 3
                # g.face_edges = 3

                # tombstone:
                g.enscribed_perimeter_ratio = np.cos(np.pi/4)
                g.circumscribed_face_ratio = 1.0
                g.enscribed_face_ratio = np.cos(np.pi/4)
                g.top_face_ratio = g.enscribed_face_ratio
                g.adjacent_face_ratio = g.enscribed_face_ratio
                g.perimeter_edges = 4
                g.face_edges = 4
                g.adjacent_faces = 0
            case 6:
                g.enscribed_perimeter_ratio = np.cos(np.pi/4)
                g.circumscribed_face_ratio = 1.0
                g.enscribed_face_ratio = np.cos(np.pi/4)
                g.top_face_ratio = g.enscribed_face_ratio
                g.adjacent_face_ratio = g.enscribed_face_ratio
                g.perimeter_edges = 4
                g.face_edges = 4
                g.adjacent_faces = 0
            case 8:
                g.enscribed_perimeter_ratio = np.cos(np.pi/6)
                g.circumscribed_face_ratio = 1.0
                g.enscribed_face_ratio = np.cos(np.pi/3)
                g.top_face_ratio = g.enscribed_face_ratio
                g.adjacent_face_ratio = g.enscribed_face_ratio
                g.perimeter_edges = 6
                g.face_edges = 3
                g.adjacent_faces = 0
            case 10:
                g.enscribed_perimeter_ratio = 0.747
                g.circumscribed_face_ratio = 0.725
                g.enscribed_face_ratio = 0.365 
                g.top_face_ratio = 0.5 #force bigger to account for offset
                g.adjacent_face_ratio = 0.95 #want to include both edge numbers
                g.perimeter_edges = 6
                g.face_edges = 4
                g.adjacent_faces = 2
                g.adjacent_face_angle = 50
            case 12:
                g.enscribed_perimeter_ratio = np.cos(np.pi/10)
                g.circumscribed_face_ratio = 0.618
                g.enscribed_face_ratio = 0.500
                g.top_face_ratio = g.enscribed_face_ratio
                g.adjacent_face_ratio = g.enscribed_perimeter_ratio
                g.perimeter_edges = 10
                g.face_edges = 5
                g.adjacent_faces = 5
                g.adjacent_face_angle = 72
            case 20:
                g.enscribed_perimeter_ratio = np.cos(np.pi/6)
                g.circumscribed_face_ratio = 0.619
                g.enscribed_face_ratio = 0.313
                g.top_face_ratio = g.enscribed_face_ratio
                g.adjacent_face_ratio = 0.75 #try to include 3 close faces but reject other 6 high angle faces
                g.perimeter_edges = 6
                g.face_edges = 3
                g.adjacent_faces = 3
                g.adjacent_face_angle = 42
            case _:
                raise ValueError("Unsupported Die Rank for Default Geometry")
        return g