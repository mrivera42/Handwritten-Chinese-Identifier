//
//  ImagePicker.swift
//  HanziReader
//
//  Created by Max Rivera on 5/2/22.
//

import SwiftUI
import PhotosUI
import UIKit

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var isPresented: Bool
    @Binding var selectedImage: UIImage
    @Binding var isShowingSelector: Bool
    @Binding var sourceType: String
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: ImagePicker
        init(parent: ImagePicker) {
            self.parent = parent
        }
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let cameraImage = info[.editedImage] as? UIImage {
                
                self.parent.selectedImage = cameraImage
            }
            self.parent.isShowingSelector = false
            self.parent.isPresented = false
        }
        
    }
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let controller = UIImagePickerController()
        if (sourceType == "camera") {
            controller.sourceType = .camera
        } else {
            controller.sourceType = .photoLibrary
        }
        controller.allowsEditing = true
        
        controller.delegate = context.coordinator
        
        return controller
    }
    
    func makeCoordinator() -> Coordinator {
        return Coordinator(parent:self)
    }
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {
        // leave blank
    }
    
    
}
