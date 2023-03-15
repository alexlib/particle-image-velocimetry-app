from openpiv import tools, pyprocess
import plotly.graph_objects as go
import imageio.v3 as iio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from io import StringIO


from viktor import ViktorController
    

from viktor.views import (
    PlotlyResult,
    ImageResult,
    WebResult,
    PlotlyView,
    ImageView,
    WebView
)

from viktor.core import (
    UserError
)


from viktor.parametrization import (
    ViktorParametrization,
    SetParamsButton,
    IntegerField,
    NumberField,
    LineBreak,
    FileField,
    Step,
    Text,
    Tab,
)

def simplePiv(firstImage, secondImage, windowSize, overlap):
    firstPath = firstImage.copy().source
    secondPath = secondImage.copy().source
    firstImage = tools.imread(firstPath)
    secondImage = tools.imread(secondPath)

    searchSize=int(windowSize * 2)
    u, v, s2n = pyprocess.extended_search_area_piv(
        firstImage.astype(np.int32), 
        secondImage.astype(np.int32), 
        window_size=windowSize,
        overlap=overlap, 
        search_area_size=searchSize,
    )
    x, y = pyprocess.get_coordinates(image_size=firstImage.shape,
                                     search_area_size=searchSize,
                                     overlap=overlap)

    valid = s2n > np.percentile(s2n, 5)

    return x, y, u, v, valid

def maxWindowSize(params, **kwargs):
    firstPath = params.uploadStep.firstUpload.file.copy().source
    firstImage = tools.imread(firstPath)
    maxSize = int(np.max(np.array(firstImage))/4)
    return maxSize #maxSize

def maxOverlap(params, **kwargs):
    maxOverlap = int(params.quiverPlotStep.windowSize-4)
    return maxOverlap
    


class Parametrization(ViktorParametrization):
    
    uploadStep = Tab( 'Step 1 - file upload')
    
    uploadStep.text = Text(
        """
# Welcome to the openPIV demo app.

In this app you can upload images to analyze them
        """
    )

    uploadStep.firstUpload = FileField(
        'upload first picture', 
        file_types=['.bmp','.png','.jpg', '.jpeg' ]
    )

    uploadStep.lb1 = LineBreak()
    uploadStep.secondUpload = FileField(
        'upload second picture', 
        file_types=['.bmp','.png','.jpg', '.jpeg' ]
    )


    quiverPlotStep = Tab('Step 2 - Generate Quiver Plot')
    
    quiverPlotStep.windowSize = NumberField(
        'Interrogation window size',
        min=8,
        max=maxWindowSize,
        default=32,
        step=4,
    )

    quiverPlotStep.lb1 = LineBreak()

    quiverPlotStep.overlap = NumberField(
        'overlap',
        min=0,
        max=maxOverlap,
        default=0,
        step=4,
    )

    quiverPlotStep.lb2 = LineBreak()
    quiverPlotStep.arrowSize = NumberField(
        'Arrow Length Scaling',
        min=15,
        max=60,
        default=20,
        variant='slider',
    )



class Controller(ViktorController):
    viktor_enforce_field_constraints = True  # Resolves upgrade instruction https://docs.viktor.ai/sdk/upgrades#U83
    label = "ImagePIVAnalysis"
    parametrization = Parametrization

    # @ImageView("Images", duration_guess=1)
    # def displayImage(self, params, **kwargs)-> ImageResult:
    #     if not params.uploadStep.firstUpload or not params.uploadStep.secondUpload:
    #         raise UserError("Please upload and select two images")
    #     return ImageResult(params.uploadStep.firstUpload.file)


    @ImageView('openPIV Results', duration_guess=2)
    def displayResult(self, params, **kwargs) -> ImageResult:
        if not params.uploadStep.firstUpload or not params.uploadStep.secondUpload:
            raise UserError("Please upload and select two images")
        x, y, u, v, valid = simplePiv(
            params.uploadStep.firstUpload.file,
            params.uploadStep.secondUpload.file,
            int(params.quiverPlotStep.windowSize),
            int(params.quiverPlotStep.overlap)
        )

        fig, ax = plt.subplots()
        firstPath = params.uploadStep.firstUpload.file.copy().source
        firstImage = tools.imread(firstPath)
        ax.imshow(firstImage, cmap="gray", alpha=0.8, origin="upper")

        C = []
        for dispx, dispy in zip(u, v):
            A = []
            for i,j in zip(dispx,dispy):
                A.append(np.sqrt(i**2 + j**2))
            C.append(A)
        C = np.array(C)
        C = C[~np.isnan(C)]
        
        arrows = ax.quiver(x[valid], y[valid], u[valid], -v[valid], C, scale= 4*params.quiverPlotStep.arrowSize)
        fig.colorbar(arrows)
        svgData=StringIO()
        fig.savefig(svgData, format='svg')
        plt.close()
        return ImageResult(svgData)



    @WebView("What's next?", duration_guess=1)
    def whats_next(self, **kwargs):
        """Initiates the process of rendering the "What's next" tab."""
        html_path = Path(__file__).parent / "final_step.html"
        with html_path.open(encoding="utf-8") as _file:
            html_string = _file.read()
        return WebResult(html=html_string)
    
