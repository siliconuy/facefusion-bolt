
    <file>
      from collections import namedtuple
      from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict
      
      import numpy
      from numpy.typing import NDArray
      from onnxruntime import InferenceSession
      
      Scale = float
      Score = float
      Angle = int
      
      Detection = NDArray[Any]
      Prediction = NDArray[Any]
      
      BoundingBox = NDArray[Any]
      FaceLandmark5 = NDArray[Any]
      FaceLandmark68 = NDArray[Any]
      FaceLandmarkSet = TypedDict('FaceLandmarkSet',
      {
      	'5' : FaceLandmark5, #type:ignore[valid-type]
      	'5/68' : FaceLandmark5, #type:ignore[valid-type]
      	'68' : FaceLandmark68, #type:ignore[valid-type]
      	'68/5' : FaceLandmark68 #type:ignore[valid-type]
      })
      FaceScoreSet = TypedDict('FaceScoreSet',
      {
      	'detector' : Score,
      	'landmarker' : Score
      })
      Embedding = NDArray[numpy.float64]
      Gender = Literal['female', 'male']
      Age = range
      Race = Literal['white', 'black', 'latino', 'asian', 'indian', 'arabic']
      Face = namedtuple('Face',
      [
      	'bounding_box',
      	'score_set',
      	'landmark_set',
      	'angle',
      	'embedding',
      	'normed_embedding',
      	'gender',
      	'age',
      	'race'
      ])
      FaceSet = Dict[str, List[Face]]
      FaceStore = TypedDict('FaceStore',
      {
      	'static_faces' : FaceSet,
      	'reference_faces' : FaceSet
      })
      
      VisionFrame = NDArray[Any]
      Mask = NDArray[Any]
      Points = NDArray[Any]
      Distance = NDArray[Any]