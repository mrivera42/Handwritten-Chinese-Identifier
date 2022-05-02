//
//  ContentView.swift
//  HanziReader
//
//  Created by Max Rivera on 4/28/22.
//

import SwiftUI
import CoreML
import Vision

struct ContentView: View {
    @State private var image: UIImage = UIImage(imageLiteralResourceName:"placeholder")
    @State private var prediction = ""
    @State var isShowingImagePicker = false
    @State var isShowingSelector = false
    @State var sourceType = ""
    @State var showingAlert = false

    var body: some View {
            Form {
                VStack(alignment: .leading) {
                
                    Text("Hanzi Reader")
                        .bold()
                        .font(.title)
                        .padding()
                    
                    Text("Use the power of AI to read your handwritten chinese characters")
                        .font(.subheadline)
                        .padding()
                    
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .padding()
                    
                    Text("Prediction: \(prediction)")
                        .padding()
                    
                    HStack(alignment: .center) {
                        Button("Select Image") {
                            self.isShowingSelector.toggle()
                            
                            
                        }
                        .buttonStyle(.bordered)
                        .sheet(isPresented:$isShowingSelector,content: {
                            VStack {
                                Button("Camera") {
                                    self.isShowingImagePicker.toggle()
                                    self.sourceType = "camera"
                                }
                                .sheet(isPresented:$isShowingImagePicker,content: {
                                    ImagePicker(isPresented: $isShowingImagePicker, selectedImage: self.$image,isShowingSelector: $isShowingSelector,sourceType: $sourceType)
                                })
                                Button("Photo Library") {
                                    self.isShowingImagePicker.toggle()
                                    self.sourceType = "library"
                                    
                                }
                                .sheet(isPresented:$isShowingImagePicker,content: {
                                    ImagePicker(isPresented: $isShowingImagePicker, selectedImage: self.$image,isShowingSelector: $isShowingSelector,sourceType: $sourceType)
                                })
                                
                                Button("Cancel") {
                                    isShowingSelector = false
                                }
                            }
                        })
                        
                        Button("Get Prediction") {
                            if (image == UIImage(imageLiteralResourceName:"placeholder")) {
                                showingAlert = true
                            } else {
                                
                                let inputImage: UIImage = image
                                let inputCGImage: CGImage? = inputImage.cgImage
                                prediction = predict(img:(inputCGImage!))
                                
                            }
                            
                        }
                        .buttonStyle(.bordered)
                        .alert("Please select an Image",isPresented:$showingAlert) {
                            Button("Ok",role:.cancel) {}
                        }
                        
                    }
                    .padding()
                    
                    Button("Clear") {
                        image = UIImage(imageLiteralResourceName:"placeholder")
                        prediction = ""
                    }
                    .buttonStyle(.bordered)
                    .frame(maxWidth:.infinity,alignment:.center)
                }
            }
    }
    
    func predict(img: CGImage) -> String {
        let config = MLModelConfiguration()
        let model = try! coreml_model(configuration: config)
        
        // Create a Vision instance using the image classifier's model instance.
        guard let imageClassifierVisionModel = try? VNCoreMLModel(for: model.model) else {
            fatalError("App failed to create a `VNCoreMLModel` instance.")
        }
        // create request
        let imageClassificationRequest = VNCoreMLRequest(model: imageClassifierVisionModel)
        
        // create handler
        let handler = VNImageRequestHandler(cgImage: img)
        
        // try
        try! handler.perform([imageClassificationRequest])
        
        //return result
        guard let observation = imageClassificationRequest.results?.first as? VNClassificationObservation else {return "did not work"}
        return observation.identifier
    }
    
    
    
}

// just for preview, debugging, and testing
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
