<div class="markdown prose w-full break-words dark:prose-invert light"><h2>Input Optimizer</h2><p>Input Optimizer is a PyTorch code designed to optimize input images for any deep learning network based on a specific target. The code was developed in the Human and Robot Interaction Laboratory (<a href="https://taarlab.com/">Taarlab</a>).
</p>
<h3>
Optimizing Input Noise on Mark1501 Dataset</h3><p>One of the library's features is the ability to optimize input noise going through a pre-trained network on Mark1501 dataset. The following images demonstrate how the library optimizes based on a specific attribute:
</p>
<p>
<img src="https://user-images.githubusercontent.com/89969561/184866464-f4bec1cd-e6c8-4bd4-a34d-2d74261edf3d.jpg" alt="c_5_iter_999_loss_-26 289833068847656"> | 
<img src="https://user-images.githubusercontent.com/89969561/184866376-db01073d-2371-4622-a934-202641562861.jpg" alt="c_30_iter_758_loss_-29 52212142944336"> | 
<img src="https://user-images.githubusercontent.com/89969561/184866334-963671f0-c7c4-429b-a2c1-5dafee5067cc.jpg" alt="c_25_iter_448_loss_-22 735559463500977"></p><h3>Optimizing Input Noise on Pre-trained AlexNet Model</h3><p>Another feature of Input Optimizer is the ability to optimize input noise going through a pre-trained AlexNet model. The following images demonstrate how the library optimizes based on a specific attribute:

</p><p>
<img src="https://user-images.githubusercontent.com/89969561/184868238-db5f0eee-bcf0-4631-97d8-bde43d05b73c.jpg" width="270">
<img src="https://user-images.githubusercontent.com/89969561/184868271-456f9e6f-4ed2-423f-956d-a4ef90555e8d.jpg" width="270">
<img src="https://user-images.githubusercontent.com/89969561/184868408-ae11166c-4d3d-4749-ac27-d78f68343310.jpg" width="270">
</p>
<h3>Requirements</h3><p>To use Input Optimizer, you need to have the following Python packages installed:</p><pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>python</span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-python">torch
torchvision
copy
numpy
PIL
</code></div></div></pre><p>For more information, please contact the developer at <a href="mailto:farbodsiahkali80@ut.ac.ir" target="_new">farbodsiahkali80@ut.ac.ir</a> or <a href="mailto:farbodsiahkali80@gmail.com" target="_new">farbodsiahkali80@gmail.com</a>.</p></div>
