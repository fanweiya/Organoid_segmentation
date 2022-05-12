package com.baidu.paddle.lite.demo.segmentation.visual;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;

import com.baidu.paddle.lite.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


public class Visualize {
    private static final String TAG = Visualize.class.getSimpleName();
    public Map<String, Bitmap> draw(Bitmap inputImage, Tensor outputTensor,Map<String,Vector<Integer>> pre_shape){

        final int[] colors_map = {0xFF000000,0xFF00FF00,0xFFFF0000};//0xFFFF0000红，0xFF00FF00绿
        long[] output = outputTensor.getLongData();
        //float[] output = outputTensor.getFloatData();
        long[] outputShape = outputTensor.shape();
        long outputSize = 1;

        for (long s : outputShape) {
            outputSize *= s;
        }

        int[] objectColor = new int[(int)outputSize];
        for(int i=0;i<output.length;i++){
            //System.out.println("no cut "+(int)output[i]);
            objectColor[i] = colors_map[(int)output[i]];
        }

        Bitmap.Config config = inputImage.getConfig();
        Bitmap outputImage = null;
        //System.out.println(Arrays.toString(outputShape));
        if(outputShape.length==3){
            outputImage = Bitmap.createBitmap(objectColor, (int)outputShape[2], (int)outputShape[1], config);
            outputImage =revese_preprecess(outputImage,inputImage.getWidth(),inputImage.getHeight(),pre_shape);
        }
        else if (outputShape.length==4){
            //System.out.println("outputShape "+(int)outputShape[2]+" "+(int)outputShape[1]);
//            outputImage = Bitmap.createBitmap(objectColor, (int)outputShape[3], (int)outputShape[2], config);
            outputImage = Bitmap.createBitmap(objectColor, (int)outputShape[2], (int)outputShape[1], config);
            //System.out.println("outputImage"+" "+outputImage.getHeight()+" "+outputImage.getWidth());
            outputImage =revese_preprecess(outputImage,inputImage.getWidth(),inputImage.getHeight(),pre_shape);
            //outputImage = Bitmap.createScaledBitmap(outputImage, inputImage.getWidth(), inputImage.getHeight(),true);
        }
        Bitmap bmOverlay = Bitmap.createBitmap(inputImage.getWidth(), inputImage.getHeight() , Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bmOverlay);
        canvas.drawBitmap(inputImage, new Matrix(), null);

        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        //原始结果
        //paint.setAlpha(0x80); 128
        paint.setAlpha(50);
        canvas.drawBitmap(outputImage, 0, 0, paint);
        Map<String, Bitmap> resultmap = new HashMap<String, Bitmap>();
        resultmap.put("mask", outputImage);
        resultmap.put("image_mask", bmOverlay);
        return resultmap;

    }
    public Map<String, Bitmap> draw_slice(Bitmap inputImage, Vector<long[]> outputs,Vector<long[]> outputShapes, Vector<Rect> slice_bboxes,Vector<Map<String,Vector<Integer>>> pre_shapes){
    Bitmap outputImage = Bitmap.createBitmap(inputImage.getWidth(), inputImage.getHeight() , Bitmap.Config.ARGB_8888);
    Canvas outputImage_canvas = new Canvas(outputImage);
    Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    //System.out.println("outputs "+outputs.size()+" "+outputShapes.size());
    //原始结果
    //paint.setAlpha(0x80); 128
    paint.setAlpha(50);
    for(int index=0;index<slice_bboxes.size();index++){
        Rect ROI_img=slice_bboxes.get(index);
        final int[] colors_map = {0xFF000000,0xFFFF0000,0xFF00FF00};//0xFFFF0000红，0xFF00FF00绿
        long[] output = outputs.get(index);
        //float[] output = outputTensor.getFloatData();
        long[] outputShape = outputShapes.get(index);
        long outputSize = 1;

        for (long s : outputShape) {
            outputSize *= s;
        }

        int[] objectColor = new int[(int)outputSize];
        //System.out.println("outputSize "+outputSize);
        for(int i=0;i<output.length;i++){
            //System.out.println("cut "+(int)output[i]);
            objectColor[i] = colors_map[(int)output[i]];
        }

        Bitmap.Config config = inputImage.getConfig();
        //System.out.println(Arrays.toString(outputShape));
        if(outputShape.length==3){
            Bitmap tempoutputImage = Bitmap.createBitmap(objectColor, (int)outputShape[2], (int)outputShape[1], config);
            //System.out.println("tempoutputImage"+" "+tempoutputImage.getHeight()+" "+tempoutputImage.getWidth());
            tempoutputImage=revese_preprecess(tempoutputImage,ROI_img.width, ROI_img.height,pre_shapes.get(index));
            //tempoutputImage=Bitmap.createScaledBitmap(tempoutputImage, ROI_img.width, ROI_img.height,true);
            outputImage_canvas.drawBitmap(tempoutputImage, ROI_img.x, ROI_img.y, null);
        }
        else if (outputShape.length==4){
            Bitmap tempoutputImage = Bitmap.createBitmap(objectColor, (int)outputShape[2], (int)outputShape[1], config);
            //System.out.println("tempoutputImage"+" "+tempoutputImage.getHeight()+" "+tempoutputImage.getWidth());
            tempoutputImage=revese_preprecess(tempoutputImage,ROI_img.width, ROI_img.height,pre_shapes.get(index));
            //tempoutputImage=Bitmap.createScaledBitmap(tempoutputImage, ROI_img.width, ROI_img.height,true);
            outputImage_canvas.drawBitmap(tempoutputImage, ROI_img.x, ROI_img.y, null);
        }
    }
    Bitmap bmOverlay = Bitmap.createBitmap(inputImage.getWidth(), inputImage.getHeight() , Bitmap.Config.ARGB_8888);
    Canvas canvas = new Canvas(bmOverlay);
    canvas.drawBitmap(inputImage, new Matrix(), null);
    canvas.drawBitmap(outputImage, 0, 0, paint);
    Map<String, Bitmap> resultmap = new HashMap<String, Bitmap>();
    resultmap.put("mask", outputImage);
    resultmap.put("image_mask", bmOverlay);
    return resultmap;
    }
    public Bitmap revese_preprecess(Bitmap outBitmap,int inputImage_width,int inputImage_height,Map<String,Vector<Integer>> pre_shape){
        Bitmap result_outBitmap = Bitmap.createScaledBitmap(outBitmap, inputImage_width, inputImage_height,true);
        return result_outBitmap;
    }
//    public Bitmap revese_preprecess(Bitmap outBitmap,int inputImage_width,int inputImage_height,Map<String,Vector<Integer>> pre_shape){
//        Mat mask=new Mat();
//        Bitmap.Config config = outBitmap.getConfig();
//        org.opencv.android.Utils.bitmapToMat(outBitmap,mask);
//        Rect crop_roi = new Rect(0, 0, pre_shape.get("pading").get(0), pre_shape.get("pading").get(1));
//        //System.out.println("mask:"+mask.width()+" "+mask.height());
//        //System.out.println("crop_roi:"+crop_roi.width+" "+crop_roi.height);
//        mask = mask.submat(crop_roi);
//        Size sz = new Size(pre_shape.get("resize").get(0), pre_shape.get("resize").get(1));
//        //System.out.println("resize "+sz.height+" "+sz.width);
//        Imgproc.resize(mask, mask, sz,0,0,Imgproc.INTER_LINEAR);
//        //System.out.println("mask "+mask.size());
//        Bitmap scaleImage = Bitmap.createBitmap(inputImage_width, inputImage_height, config);
//        //System.out.println("scaleImage "+scaleImage.getWidth()+" "+scaleImage.getHeight());
//        org.opencv.android.Utils.matToBitmap(mask,scaleImage);
//        return scaleImage;
//    }
}
