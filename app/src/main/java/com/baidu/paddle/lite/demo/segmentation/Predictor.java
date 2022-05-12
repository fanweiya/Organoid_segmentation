package com.baidu.paddle.lite.demo.segmentation;

import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.ellipse;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.os.Environment;
import android.util.Log;

import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.PaddlePredictor;
import com.baidu.paddle.lite.PowerMode;
import com.baidu.paddle.lite.Tensor;
import com.baidu.paddle.lite.demo.segmentation.config.Config;
import com.baidu.paddle.lite.demo.segmentation.preprocess.Preprocess;
import com.baidu.paddle.lite.demo.segmentation.visual.Visualize;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Vector;

public class Predictor {
    private static final String TAG = Predictor.class.getSimpleName();
    protected Vector<String> wordLabels = new Vector<String>();
    Config config = new Config();
    public Map<String,Vector<Integer>> pre_shape=new HashMap<>();
    protected Bitmap inputImage = null;
    protected Bitmap scaledImage = null;
    protected Bitmap outputImage = null;
    protected String filename;
    protected String outputResult = "";
    protected float preprocessTime = 0;
    protected float postprocessTime = 0;
    protected String dirPath = Environment.getExternalStorageDirectory() + "/leiqiguan_Demo";
    public boolean isLoaded = false;
    public int warmupIterNum = 0;
    public int inferIterNum = 1;
    protected Context appCtx = null;
    public int cpuThreadNum = 1;
    public String cpuPowerMode = "LITE_POWER_HIGH";
    public String modelPath = "";
    public String modelName = "";
    protected PaddlePredictor paddlePredictor = null;
    protected float inferenceTime = 0;

    public Predictor() {
        super();
    }

    public boolean init(Context appCtx, String modelPath, int cpuThreadNum, String cpuPowerMode) {
        this.appCtx = appCtx;
        isLoaded = loadModel(modelPath, cpuThreadNum, cpuPowerMode);
        return isLoaded;
    }

    public boolean init(Context appCtx, Config config) {

        if (config.inputShape.length != 4) {
            Log.i(TAG, "size of input shape should be: 4");
            return false;
        }
        if (config.inputShape[0] != 1) {
            Log.i(TAG, "only one batch is supported in the image classification demo, you can use any batch size in " +
                    "your Apps!");
            return false;
        }
        if (config.inputShape[1] != 1 && config.inputShape[1] != 3) {
            Log.i(TAG, "only one/three channels are supported in the image classification demo, you can use any " +
                    "channel size in your Apps!");
            return false;
        }
        if (!config.inputColorFormat.equalsIgnoreCase("RGB") && !config.inputColorFormat.equalsIgnoreCase("BGR")) {
            Log.i(TAG, "only RGB and BGR color format is supported.");
            return false;
        }
        init(appCtx, config.modelPath, config.cpuThreadNum, config.cpuPowerMode);

        if (!isLoaded()) {
            return false;
        }
        this.config = config;

        return isLoaded;
    }


    public boolean isLoaded() {
        return paddlePredictor != null && isLoaded;
    }

    protected boolean loadLabel(String labelPath) {
        wordLabels.clear();
        // load word labels from file
        try {
            InputStream assetsInputStream = appCtx.getAssets().open(labelPath);
            int available = assetsInputStream.available();
            byte[] lines = new byte[available];
            assetsInputStream.read(lines);
            assetsInputStream.close();
            String words = new String(lines);
            String[] contents = words.split("\n");
            for (String content : contents) {
                wordLabels.add(content);
            }
            Log.i(TAG, "word label size: " + wordLabels.size());
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
            return false;
        }
        return true;
    }

    public Tensor getInput(int idx) {
        if (!isLoaded()) {
            return null;
        }
        return paddlePredictor.getInput(idx);
    }

    public Tensor getOutput(int idx) {
        if (!isLoaded()) {
            return null;
        }
        return paddlePredictor.getOutput(idx);
    }

    protected boolean loadModel(String modelPath, int cpuThreadNum, String cpuPowerMode) {
        // release model if exists
        releaseModel();

        // load model
        if (modelPath.isEmpty()) {
            return false;
        }
        String realPath = modelPath;
        if (!modelPath.substring(0, 1).equals("/")) {
            // read model files from custom file_paths if the first character of mode file_paths is '/'
            // otherwise copy model to cache from assets
            realPath = appCtx.getCacheDir() + "/" + modelPath;
            Utils.copyDirectoryFromAssets(appCtx, modelPath, realPath);
        }
        if (realPath.isEmpty()) {
            return false;
        }
        MobileConfig config = new MobileConfig();
        config.setModelFromFile(realPath + File.separator + "pp_liteseg_stdc1_A_B_C_sigle_data_agg_512x512.nb");
        config.setThreads(cpuThreadNum);
        if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_HIGH")) {
            config.setPowerMode(PowerMode.LITE_POWER_HIGH);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_LOW")) {
            config.setPowerMode(PowerMode.LITE_POWER_LOW);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_FULL")) {
            config.setPowerMode(PowerMode.LITE_POWER_FULL);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_NO_BIND")) {
            config.setPowerMode(PowerMode.LITE_POWER_NO_BIND);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_RAND_HIGH")) {
            config.setPowerMode(PowerMode.LITE_POWER_RAND_HIGH);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_RAND_LOW")) {
            config.setPowerMode(PowerMode.LITE_POWER_RAND_LOW);
        } else {
            Log.e(TAG, "unknown cpu power mode!");
            return false;
        }
        paddlePredictor = PaddlePredictor.createPaddlePredictor(config);
        this.cpuThreadNum = cpuThreadNum;
        this.cpuPowerMode = cpuPowerMode;
        this.modelPath = realPath;
        this.modelName = realPath.substring(realPath.lastIndexOf("/") + 1);
        return true;
    }

    public boolean runModel() {
        if (!isLoaded()) {
            return false;
        }
        // warm up
        for (int i = 0; i < warmupIterNum; i++){
            paddlePredictor.run();
        }
        //System.out.println("一次推理");
        // inference
        Date start = new Date();
        for (int i = 0; i < inferIterNum; i++) {
            paddlePredictor.run();
        }
        Date end = new Date();
        inferenceTime = (end.getTime() - start.getTime()) / (float) inferIterNum;
        return true;
    }

    public boolean runModel(Bitmap image) {
        setInputImage(image);
        return runModel();
    }
    void saveBitmap(String picName, Bitmap resultbimap) {
        File dir = new File(dirPath);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        File f = new File(dirPath, picName);
        if (f.exists()) {
            f.delete();
        }
        try {
            FileOutputStream out = new FileOutputStream(f);
            resultbimap.compress(Bitmap.CompressFormat.PNG, 100, out);
            out.flush();
            out.close();
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
//    public Mat Watershed(Mat img){
//        //watershed algorithm
//        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGRA2BGR);
//        Mat gray = new Mat();
//        Mat thresh = new Mat();
//        double ret;
//        Mat  kernel;
//        Mat opening = new Mat();
//        Mat sure_bg = new Mat();
//        Point anchor = new Point(-1, -1); // default value
//        Mat dist_transform = new Mat();
//        Mat sure_fg = new Mat();
//        Mat unknown = new Mat();
//        Mat markers = new Mat();
//        Scalar sc = new Scalar(-1);
//
//        Imgproc.cvtColor(img, gray, Imgproc.COLOR_RGB2GRAY);
//        ret = Imgproc.threshold(gray, thresh, 0 ,255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
//
//        // remove noise
//        kernel = Mat.ones(3, 3, CvType.CV_8UC(1));
//        Imgproc.morphologyEx(thresh, opening, Imgproc.MORPH_OPEN, kernel, anchor, 1);
//
//        // Find the sure background region
//        Imgproc.dilate(opening, sure_bg, kernel, anchor, 8);
//
//        // Find the sure foreground region
//        Imgproc.distanceTransform(opening, dist_transform,Imgproc.DIST_L2, 5);
//        Core.MinMaxLocResult mmr= Core.minMaxLoc(dist_transform);
//        ret = Imgproc.threshold(dist_transform, sure_fg, 0.7 * mmr.maxVal ,255, 0);
//        sure_fg.assignTo(sure_fg, CvType.CV_8UC(sure_fg.channels()));
//        //Log.i("value", String.valueOf(0.7 * mmr.maxVal));
//
//        // Find the unknown region
//        Core.subtract(sure_bg, sure_fg, unknown);
//
//        // Label the foreground objects
//        ret = Imgproc.connectedComponents(sure_fg, markers);
//
//        // Add one to all labels so that sure background is not 0, but 1
//        Core.subtract(markers, sc, markers);
//        //for(int i=0; i <markers.rows(); i++){
//
//        //  for(int j=0; j < markers.cols(); j++){
//        //    markers.put(i, j, markers.get(i, j)[0] + 1);
//        //}
//        //}
//
//        // Label the unknown region as 0
//        for(int i=0; i <markers.rows(); i++){
//
//            for(int j=0; j < markers.cols(); j++){
//                if(unknown.get(i, j)[0] == 255){
//                    markers.put(i, j, 0);
//                }
//            }
//        }
//
//        Imgproc.watershed(img, markers);
//
//        for(int i=0; i <markers.rows(); i++){
//
//            for(int j=0; j < markers.cols(); j++){
//                if(markers.get(i, j)[0] == 1){ //1 is sure background (use -1 for sure boundary region)
//                    img.put(i, j, 0,0,0);
//                }
//            }
//        }
//
//        return img;
//    }
    public List<MatOfPoint> flitercontours(List<MatOfPoint> A, List<MatOfPoint> B){
        //判断B轮廓的点是否在A的轮廓中，如果在，则删除A中的轮廓，返回A+B
        //System.out.println("A.size()="+A.size()+" B.size()="+B.size());
        for(int i=0;i<B.size();i++){
            MatOfPoint b=B.get(i);
            for(int j=0;j<A.size();j++){
                MatOfPoint a=A.get(j);
                MatOfPoint2f a2f=new MatOfPoint2f(a.toArray());
                if(Imgproc.pointPolygonTest(a2f,b.toArray()[0],false)>=0){
                    A.remove(j);
                }
            }
        }
        //System.out.println("A.size()="+A.size()+" B.size()="+B.size());
        List<MatOfPoint> C=new ArrayList<MatOfPoint>();
        C.addAll(A);
        C.addAll(B);
        return C;
    }
    public List<MatOfPoint> fastGetCouter(Mat markers,int CouterNUM){
        List<MatOfPoint> end_contours = new ArrayList<>();
        List<List<Integer>> partsLists = new ArrayList<>();
        List<Integer> temp = new ArrayList<>();
        for(int obj = 1; obj < CouterNUM + 1; obj++)
        {
            temp.add(obj);
            if(obj%255==0){
                partsLists.add(temp);
                temp=new ArrayList<>();
            }
        }
        partsLists.add(temp);
//        int testzero = 0;
//        for(List<Integer> l:partsLists){
//            String temp_str="";
//            String temp_str1="";
//            for(int i:l){
//                temp_str=temp_str+ i +",";
//                temp_str1=temp_str1+ (i-255*testzero) +" ";
//            }
//            testzero++;
//            System.out.println(temp_str+"\n"+temp_str1);
//        }
        for(int z=0;z<partsLists.size();z++){
            List<Integer> partlist=partsLists.get(z);
            Mat dst = Mat.zeros(markers.size(), CvType.CV_8UC1);
            byte[] dstData = new byte[(int) (dst.total() * dst.channels())];
            dst.get(0, 0, dstData);
            // Fill labeled objects with random colors
            int[] markersData = new int[(int) (markers.total() * markers.channels())];
            markers.get(0, 0, markersData);
            for (int i = 0; i < markers.rows(); i++) {
                for (int j = 0; j < markers.cols(); j++) {
                    int index = markersData[i * markers.cols() + j];
                    //System.out.println(index+" "+partlist.contains(index)+" z="+z+" index-255*z="+(index-255*z));
                    if (partlist.contains(index)) {
                        dstData[i * dst.cols() + j] = (byte) (index-255*z);
                    } else {
                        dstData[i * dst.cols() + j] = 0;
                    }
                }
            }
            dst.put(0, 0, dstData);
            for(int objj = 1; objj < 256; objj++) {
                Mat objMask=new Mat();
                Core.inRange(dst, new Scalar(objj), new Scalar(objj), objMask);
                List<MatOfPoint> temp_contours = new ArrayList<>();
                Imgproc.findContours(objMask,temp_contours,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
                if(temp_contours.size()>0){
                    MatOfPoint temp_tours=temp_contours.get(0);
                    end_contours.add(temp_tours);
                }
            }
        }
        return end_contours;
    }
    public List<MatOfPoint> watershed_process(Mat imgResult,double thrad,double itereation){
        // Create binary image from source image
        //Imgproc.cvtColor(imgResult, imgResult, Imgproc.COLOR_BGRA2BGR);
        Date start=new Date();
        Mat bw = new Mat();
        Imgproc.cvtColor(imgResult, bw, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(bw, bw, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
        List<MatOfPoint> temp_contours1 = new ArrayList<>();
        Imgproc.findContours(bw,temp_contours1,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
        //Imgcodecs.imwrite(dirPath+"/"+"bw.jpg",bw);
        // Perform the distance transform algorithm
        Mat dist = new Mat();
        Imgproc.distanceTransform(bw, dist, Imgproc.DIST_L2, 3);
        //Imgcodecs.imwrite(dirPath+"/"+"dist.jpg",dist);
        // Normalize the distance image for range = {0.0, 1.0}
        // so we can visualize and threshold it
        Core.normalize(dist, dist, 0.000, 1.000, Core.NORM_MINMAX);
//        Mat distDisplayScaled = new Mat();
//        Core.multiply(dist, new Scalar(255), distDisplayScaled);
//        Mat distDisplay = new Mat();
//        distDisplayScaled.convertTo(distDisplay, CvType.CV_8U);
        // Threshold to obtain the peaks
        // This will be the markers for the foreground objects
        Imgproc.threshold(dist, dist, thrad, 1.0, Imgproc.THRESH_BINARY);
        // Dilate a bit the dist image
        //Mat kernel1 = Mat.ones(7, 7, CvType.CV_8U);
        Mat kernel2 = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(7, 7));
//        Imgproc.dilate(dist, dist, kernel1);
        Imgproc.erode(dist,dist,kernel2,new Point(-1,-1),(int)itereation);
        //Imgproc.morphologyEx(dist, dist, Imgproc.MORPH_CLOSE, kernel2, new Point(-1, -1), 1);
//        Mat distDisplay2 = new Mat();
//        dist.convertTo(distDisplay2, CvType.CV_8U);
//        Core.multiply(distDisplay2, new Scalar(255), distDisplay2);
        // Create the CV_8U version of the distance image
        // It is needed for findContours()
        Mat dist_8u = new Mat();
        dist.convertTo(dist_8u, CvType.CV_8U);
        // Find total markers
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(dist_8u, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        // Create the marker image for the watershed algorithm
        Mat markers = Mat.zeros(dist.size(), CvType.CV_32S);
        // Draw the foreground markers
        contours=flitercontours(temp_contours1,contours);
        for (int i = 0; i < contours.size(); i++) {
            Imgproc.drawContours(markers, contours, i, new Scalar(i + 1), -1);
        }
        // Draw the background marker
//        Mat markersScaled = new Mat();
//        markers.convertTo(markersScaled, CvType.CV_32F);
//        Core.normalize(markersScaled, markersScaled, 0.0, 255.0, Core.NORM_MINMAX);
//        Imgproc.circle(markersScaled, new Point(5, 5), 3, new Scalar(255, 255, 255), -1);
//        Mat markersDisplay = new Mat();
//        markersScaled.convertTo(markersDisplay, CvType.CV_8U);
        Imgproc.circle(markers, new Point(5, 5), 3, new Scalar(255, 255, 255), -1);
        // Perform the watershed algorithm
        Imgproc.watershed(imgResult, markers);

//        Date end1=new Date();
//        System.out.println("watshed finneh "+(float)(end1.getTime()-start.getTime()));
//        List<MatOfPoint> end_contours = new ArrayList<>();
//        for(int obj = 1; obj < contours.size() + 1; obj++)
//        {
//            Mat dst = Mat.zeros(markers.size(), CvType.CV_8UC1);
//            byte[] dstData = new byte[(int) (dst.total() * dst.channels())];
//            dst.get(0, 0, dstData);
//            // Fill labeled objects with random colors
//            int[] markersData = new int[(int) (markers.total() * markers.channels())];
//            markers.get(0, 0, markersData);
//            for (int i = 0; i < markers.rows(); i++) {
//                for (int j = 0; j < markers.cols(); j++) {
//                    int index = markersData[i * markers.cols() + j];
//                    if (index== obj && index <= contours.size()) {
//                        dstData[i * dst.cols() + j] = (byte) 255;
//                    } else {
//                        dstData[i * dst.cols() + j] = 0;
//                    }
//                }
//            }
//            dst.put(0, 0, dstData);
//            //Mat graymat=new Mat();
//            List<MatOfPoint> temp_contours = new ArrayList<>();
//            //Imgproc.cvtColor(dst, graymat, Imgproc.COLOR_BGR2GRAY);
//            Imgproc.findContours(dst,temp_contours,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
//            if(temp_contours.size()>0){
//                MatOfPoint temp_tours=temp_contours.get(0);
//                end_contours.add(temp_tours);
//            }
//        }
        Date start2=new Date();
//        Mat mark = Mat.zeros(markers.size(), CvType.CV_8U);
//        markers.convertTo(mark, CvType.CV_8UC1);
//        Core.bitwise_not(mark, mark);
//        // imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
//        // image looks like at that point
//        // Generate random colors

//        Random rng = new Random(12345);
//        List<Scalar> colors = new ArrayList<>(contours.size());
//        for (int i = 0; i < contours.size(); i++) {
//            int b = rng.nextInt(256);
//            int g = rng.nextInt(256);
//            int r = rng.nextInt(256);
//            colors.add(new Scalar(b, g, r));
//        }
//        // Create the result image
//        Mat dst = Mat.zeros(markers.size(), CvType.CV_8UC3);
//        byte[] dstData = new byte[(int) (dst.total() * dst.channels())];
//        dst.get(0, 0, dstData);
//        // Fill labeled objects with random colors
//        int[] markersData = new int[(int) (markers.total() * markers.channels())];
//        markers.get(0, 0, markersData);
//        for (int i = 0; i < markers.rows(); i++) {
//            for (int j = 0; j < markers.cols(); j++) {
//                int index = markersData[i * markers.cols() + j];
//                if (index > 0 && index <= contours.size()) {
//                    dstData[(i * dst.cols() + j) * 3 + 0] = (byte) colors.get(index - 1).val[0];
//                    dstData[(i * dst.cols() + j) * 3 + 1] = (byte) colors.get(index - 1).val[1];
//                    dstData[(i * dst.cols() + j) * 3 + 2] = (byte) colors.get(index - 1).val[2];
//                } else {
//                    dstData[(i * dst.cols() + j) * 3 + 0] = 0;
//                    dstData[(i * dst.cols() + j) * 3 + 1] = 0;
//                    dstData[(i * dst.cols() + j) * 3 + 2] = 0;
//                }
//            }
//        }
//        dst.put(0, 0, dstData);
//        cvtColor(drawimage,drawimage,Imgproc.COLOR_BGRA2BGR);
//        Core.addWeighted(drawimage, 0.5, dst, 0.5, 0.0, drawimage);

        List<MatOfPoint> end_contours = new ArrayList<>();
        end_contours=fastGetCouter(markers,contours.size());
        // Visualize the final image
        Date end3=new Date();
        System.out.println("get counter cost time2 "+(float)(end3.getTime()-start2.getTime()));
//        return dst;
        return end_contours;
    }
    public Vector<Rect> getboxs(int image_height, int image_width, int slice_height, int slice_width, float overlap_height_ratio, float overlap_width_ratio) {
        Vector<Rect> slice_bboxes =new Vector<>();
        int y_max = 0;
        int y_min = 0;
        int y_overlap = Math.round(overlap_height_ratio * slice_height);
        int x_overlap = Math.round(overlap_width_ratio * slice_width);
        while (y_max < image_height)
        {
            int x_min = 0;
            int x_max = 0;
            y_max = y_min + slice_height;
            while (x_max < image_width)
            {
                x_max = x_min + slice_width;
                if ((y_max > image_height) || (x_max > image_width)) {
                int xmax = Math.min(image_width, x_max);
                int ymax = Math.min(image_height, y_max);
                int xmin = Math.max(0, xmax - slice_width);
                int ymin = Math.max(0, ymax - slice_height);
                Rect temp = new Rect();
                temp.x=xmin;temp.y=ymin;temp.height=(ymax-ymin);temp.width=(xmax-xmin);
                //temp.add(ymin);temp.add(xmax);temp.add(ymax);
                slice_bboxes.add(temp);
            }
			else {
			    Rect temp = new Rect();
			    temp.x=x_min;temp.y=y_min;temp.height=(y_max-y_min);temp.width=(x_max-x_min);
			    //temp.add(x_min);temp.add(y_min);temp.add(x_max);temp.add(y_max);
                slice_bboxes.add(temp);
            }
                x_min = x_max - x_overlap;
            }
            y_min = y_max - y_overlap;
        }
        return slice_bboxes;
    }
    public boolean runModel(Preprocess preprocess, Visualize visualize,int BG_TH,double Area_TH,double SHARP_TH,double CUT_NUM,double OVERLAP_NUM,boolean NOCUT_mode,double postthrd,double postitertion) {
        if (inputImage == null) {
            return false;
        }
        if(OVERLAP_NUM==1.0){
            System.out.println("overlap不能为1");
            OVERLAP_NUM=0.85;
            System.out.println("overlap设置为 "+0.85);

        }
        Map<String, Bitmap> model_result=new HashMap<>();
        if(NOCUT_mode){
            // set input shape
            Date start = new Date();
            System.out.println("未切片");
            Tensor inputTensor = getInput(0);
            inputTensor.resize(config.inputShape);

            // pre-process image
            preprocess.init(config);
            preprocess.to_array(scaledImage);
            // feed input tensor with pre-processed data
            inputTensor.setData(preprocess.inputData);
            Date end = new Date();
            preprocessTime = (float) (end.getTime() - start.getTime());
            // inference
            runModel();
            Tensor outputTensor = getOutput(0);
            //post-process
            model_result=visualize.draw(inputImage, outputTensor,pre_shape);
            end = new Date();
            postprocessTime = (float) (end.getTime() - start.getTime());
            System.out.println("model cost time "+postprocessTime);
        }
        else {
            Map<String, Bitmap> model_result_origin=new HashMap<>();
            Map<String, Bitmap> model_result_cut=new HashMap<>();
            // set input shape
            Date start = new Date();
            System.out.println("整体分割");
            Tensor inputTensor = getInput(0);
            inputTensor.resize(config.inputShape);
            // pre-process image
            preprocess.init(config);
            preprocess.to_array(scaledImage);
            // feed input tensor with pre-processed data
            inputTensor.setData(preprocess.inputData);
            // inference
            runModel();
            Tensor outputTensor = getOutput(0);
            //post-proces
            model_result_origin=visualize.draw(inputImage, outputTensor,pre_shape);
            Vector<Rect> slice_bboxes = getboxs(inputImage.getHeight(), inputImage.getWidth(),  (int)(inputImage.getHeight()/CUT_NUM), (int)(inputImage.getWidth()/CUT_NUM), (float)OVERLAP_NUM,(float)OVERLAP_NUM);
            Vector<long[]> outputs =new Vector<>();
            Vector<long[]> outputShapes =new Vector<>();
            Vector<Map<String,Vector<Integer>>> pre_shapes=new Vector<>();
            //System.out.println("this.config.inputShape"+this.config.inputShape[2]+" "+this.config.inputShape[3]);
            System.out.println("切片分割 "+inputImage.getHeight()+" "+inputImage.getWidth()+" "+slice_bboxes.size());
            for(int i=0;i<slice_bboxes.size();i++){
                Rect img_ROI=slice_bboxes.get(i);
                //System.out.println("slice_bboxes"+" "+img_ROI.x+" "+img_ROI.y+" "+img_ROI.width+" "+img_ROI.height);
                Bitmap CUTImage=Bitmap.createBitmap(inputImage,img_ROI.x, img_ROI.y, img_ROI.width, img_ROI.height,null, false);
                //System.out.println(i+" "+CUTImage.getHeight()+" "+CUTImage.getWidth());
                CUTImage = ReizeByLongandPading(CUTImage,this.config.inputShape);
                //CUTImage = Bitmap.createScaledBitmap(CUTImage, (int) this.config.inputShape[3], (int) this.config.inputShape[2], true);
                pre_shapes.insertElementAt(pre_shape,i);
                //System.out.println(i+" "+CUTImage.getHeight()+" "+CUTImage.getWidth());
                // set input shape
                // pre-process image
                inputTensor = getInput(0);
                inputTensor.resize(config.inputShape);
                preprocess.init(config);
                preprocess.to_array(CUTImage);
                // feed input tensor with pre-processed data
                inputTensor.setData(preprocess.inputData);
                Date end = new Date();
                preprocessTime = (float) (end.getTime() - start.getTime());
                // inference
                runModel();
                outputTensor = getOutput(0);
                long[] output = outputTensor.getLongData();
                long[] outputShape = outputTensor.shape();
                outputs.add(output.clone());
                outputShapes.add(outputShape.clone());
                end = new Date();
                postprocessTime = (float) (end.getTime() - start.getTime());
            }
            model_result_cut=visualize.draw_slice(inputImage, outputs,outputShapes,slice_bboxes,pre_shapes);
            Paint maskpaint = new Paint(Paint.ANTI_ALIAS_FLAG);
            //maskpaint.setAlpha(200);
            Bitmap sumask = Bitmap.createBitmap(inputImage.getWidth(), inputImage.getHeight() , Bitmap.Config.ARGB_8888);
            Bitmap bfOverlay = Bitmap.createBitmap(inputImage.getWidth(), inputImage.getHeight() , Bitmap.Config.ARGB_8888);
            Canvas maskcanvas = new Canvas(sumask);
            maskcanvas.drawBitmap(model_result_origin.get("mask"), 0, 0, maskpaint);
            maskpaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.LIGHTEN));
            maskcanvas.drawBitmap(model_result_cut.get("mask"), 0, 0, maskpaint);
            Canvas bm_ovlycanvas = new Canvas(bfOverlay);
            Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
            paint.setAlpha(50);
            bm_ovlycanvas.drawBitmap(inputImage, new Matrix(), null);
            bm_ovlycanvas.drawBitmap(sumask,0, 0, paint);
            model_result.put("mask",sumask);
            model_result.put("image_mask",bfOverlay);
            Date sumend = new Date();
            System.out.println("model cost time "+(float) (sumend.getTime()-start.getTime()));
        }
        //openvcv
        Mat maskgraymat=new Mat();
        Mat originimgmat = new Mat(inputImage.getHeight(), inputImage.getWidth(), CvType.CV_8UC3);
        Mat maskImagemat=new Mat();
        Mat image_maskImagemat=new Mat();
        Mat maskresmat = new Mat(inputImage.getHeight(), inputImage.getWidth(), CvType.CV_8UC3);//crop restmat
        Mat maskresgrayamt=new Mat();
        Mat maskresBinmat = new Mat();
        Mat origingraymat = new Mat();
        Bitmap maskbitmap=model_result.get("mask");
        Bitmap image_maskbitmap=model_result.get("image_mask");
        org.opencv.android.Utils.bitmapToMat(maskbitmap,maskImagemat);
        org.opencv.android.Utils.bitmapToMat(image_maskbitmap,image_maskImagemat);
        //List<MatOfPoint> calss2_contours =watershed_process(calss2_Imagemat);
        org.opencv.android.Utils.bitmapToMat(inputImage,originimgmat);
        Imgproc.cvtColor(maskImagemat, maskgraymat, Imgproc.COLOR_BGRA2GRAY);
        Imgproc.cvtColor(originimgmat, origingraymat, Imgproc.COLOR_BGRA2GRAY);
        Core.bitwise_and(originimgmat,originimgmat,maskresmat,maskgraymat);//crop分割结果
        Imgproc.cvtColor(maskresmat, maskgraymat, Imgproc.COLOR_BGRA2GRAY);
        Imgproc.threshold(maskgraymat, maskgraymat, 0, 255, Imgproc.THRESH_BINARY);
        Imgproc.cvtColor(maskgraymat, maskgraymat, Imgproc.COLOR_GRAY2BGR);
        //Imgproc.cvtColor(maskresmat, maskresmat, Imgproc.COLOR_RGB2BGR);
        Imgcodecs.imwrite(dirPath+"/"+"maskImagemat.jpg",maskImagemat);
//        Imgcodecs.imwrite(dirPath+"/"+"maskresmat.jpg",maskresmat);
//        Imgcodecs.imwrite(dirPath+"/"+"maskgraymat.jpg",maskgraymat);
        List<MatOfPoint> All_contours = new ArrayList<>();
//        Imgproc.findContours(maskgraymat,All_contours,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
        Date watershed_start = new Date();
        if(postthrd>1 || postthrd<0){
            System.out.println("postthrd not <1");
            postthrd=0.5;
        }
        if(postitertion<1){
            System.out.println("postitertion not <1");
            postitertion=1;
        }
        All_contours=watershed_process(maskgraymat,postthrd,postitertion);
        Date watershed_end = new Date();
        System.out.println("watershed cost time "+(float)(watershed_end.getTime()-watershed_start.getTime()));
        //All_contours=watershed_process(maskImagemat);
//        if(NOCUT_mode){
//            Imgproc.findContours(maskgraymat,All_contours,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
//        }else {
//            //预处理做切分
//            Mat gray_bin_mat=new Mat();
//            Imgcodecs.imwrite(dirPath+"/"+"maskgraymat.jpg",maskgraymat);
//            Imgproc.threshold(maskgraymat, gray_bin_mat, 0, 255, Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU);
//            Imgcodecs.imwrite(dirPath+"/"+"gray_bin_mat.jpg",gray_bin_mat);
//            Mat structImage = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(7,7));
//            Imgproc.morphologyEx(gray_bin_mat,gray_bin_mat,Imgproc.MORPH_CLOSE,structImage);
//            Imgproc.findContours(gray_bin_mat,All_contours,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
//        }
//        Imgproc.dilate(gray_bin_mat,gray_bin_mat,structImage,new Point(0,0));
//        Imgproc.erode(gray_bin_mat,gray_bin_mat,structImage,new Point(0,0));
        double SharpThred = SHARP_TH;
        double AreaThred = Area_TH;
        int num=0;
        String resultext=" ";
        int BG_THresh=BG_TH;
        System.out.println("BG_THresh "+BG_THresh);
        System.out.println("AreaThred "+AreaThred);
        System.out.println("SharpThred "+SharpThred);
        System.out.println("CUT_NUM "+CUT_NUM);
        System.out.println("OVERLAP_NUM "+OVERLAP_NUM);
        Imgproc.cvtColor(maskresmat, maskresgrayamt, Imgproc.COLOR_BGRA2GRAY);
        Imgproc.threshold(maskresgrayamt, maskresBinmat, BG_THresh, 255, Imgproc.THRESH_BINARY);
        int thickness=2;
        double font_scale=1;
        for(int i=0;i<All_contours.size();i++){
            MatOfPoint contour=All_contours.get(i);
            double area = Imgproc.contourArea(contour);
            MatOfPoint2f dst = new MatOfPoint2f();
            contour.convertTo(dst, CvType.CV_32F);
            double zhouchang=Imgproc.arcLength(dst,true);
            if(area > AreaThred){
                Rect rect = Imgproc.boundingRect(contour);
                Mat cropped = maskresBinmat.submat(rect);
                Mat grayimg_cropped = origingraymat.submat(rect);
                //Imgcodecs.imwrite(dirPath+"/"+i+".jpg",cropped);
                Mat destination=new Mat();
                Imgproc.Laplacian(grayimg_cropped, destination,3);
                MatOfDouble median = new MatOfDouble();
                MatOfDouble std= new MatOfDouble();
                Core.meanStdDev(destination, median , std);
                double variance = Math.pow(std.get(0,0)[0], 2);
                System.out.println(num+1+" variance: "+variance);
                if(variance>SHARP_TH){
                    int witeysum=Core.countNonZero(cropped);
                    //System.out.println(cropped.channels()+" "+cropped.rows()+" "+cropped.rows()+" "+witeysum+" "+area+" "+(rect.height*rect.width));
                    double roundness=(Math.PI*4*area)/(zhouchang*zhouchang);
                    float touguanglv=witeysum*100f/(rect.height*rect.width);
                    int diam=Math.max(rect.width,rect.height);
                    double eccentricity=0;
                    System.out.println("dst.total "+dst.total());
                    if(dst.total()>5) {
                        RotatedRect elipse = Imgproc.fitEllipse(dst);
                        //System.out.println(elipse.size.height+" "+elipse.size.width);
                        eccentricity = Math.sqrt(1 - Math.pow(Math.min(elipse.size.height, elipse.size.width) / Math.max(elipse.size.height, elipse.size.width), 2));
                    }
                    //.out.println(eccentricity);
                    //int fontScale=(int) (Math.round(0.00002 * (rect.width + rect.height) / 2) + 1);
                    Imgproc.drawContours(image_maskImagemat, All_contours,i , new Scalar(255, 255, 0), thickness,16);
                    Imgproc.putText(image_maskImagemat,String.valueOf(num+1),new Point(rect.x+(int)(rect.width/2),rect.y+(int)(rect.height/2)),Imgproc.FONT_HERSHEY_SIMPLEX,font_scale,new Scalar(255, 255, 0),thickness,16);
                    resultext = resultext+"细胞 "+String.valueOf(num+1)+": 直径："+String.valueOf(diam)+"pixel 圆度："+String.format("%.2f",(float)roundness)+" 偏心率："+String.format("%.2f",(float)eccentricity)+" 面积："+String.valueOf(area)+" pixel"+" 细胞团中央透光率:"
                            +String.format("%.2f",touguanglv)+"% "+"清晰度："+String.format("%.2f",variance)+" \n";
                    num=num+1;
                }
            }
        }
//        System.out.println(Core.countNonZero(maskresmat));
        //Imgcodecs.imwrite(dirPath+"/"+"test_result.jpg",image_maskImagemat);
        Bitmap newbit=inputImage.copy(Bitmap.Config.ARGB_8888, true);
        org.opencv.android.Utils.matToBitmap(image_maskImagemat,newbit);
        this.outputImage=newbit;
        outputResult = String.valueOf("发现 "+num+"个细胞; "+"\n"+resultext);
        System.out.println(getPicNameFromPath(config.imagePath));
        filename=getPicNameFromPath(config.imagePath);
        //saveBitmap(getPicNameFromPath(config.imagePath),newbit);
        return true;
    }
    public boolean focusDetect(Mat grayimg,int diff_sum_thre){
        int diff = 0;
        int diff_thre = 20;
        //int diff_sum_thre = 1000;
        for (int i = grayimg.rows() / 10; i < grayimg.rows(); i += grayimg.rows() / 10){
            for (int j = 0; j < grayimg.cols() - 1; j++){
                if (Math.abs(grayimg.get(i,j + 1)[0] - grayimg.get(i,j)[0])>diff_thre)
                    diff += Math.abs(grayimg.get(i,j + 1)[0] - grayimg.get(i,j)[0]);
            }
            System.out.println("diff "+diff);
        }
        boolean res = true;
        if (diff < diff_sum_thre) {
            System.out.println("the focus might be wrong!");
            res = false;
        }
        return res;
    }
    public static String getPicNameFromPath(String picturePath){
        String temp[] = picturePath.replaceAll("\\\\","/").split("/");
        String fileName = "";
        if(temp.length > 1){
            fileName = temp[temp.length - 1];
        }
        return fileName;
    }

    public void releaseModel() {
        paddlePredictor = null;
        isLoaded = false;
        cpuThreadNum = 1;
        cpuPowerMode = "LITE_POWER_HIGH";
        modelPath = "";
        modelName = "";
    }

    public void setConfig(Config config){
        this.config = config;
    }

    public Bitmap inputImage() {
        return inputImage;
    }

    public Bitmap outputImage() {
        return outputImage;
    }

    public String outputResult() {
        return outputResult;
    }

    public float preprocessTime() {
        return preprocessTime;
    }

    public float postprocessTime() {
        return postprocessTime;
    }

    public String modelPath() {
        return modelPath;
    }

    public String modelName() {
        return modelName;
    }

    public int cpuThreadNum() {
        return cpuThreadNum;
    }

    public String cpuPowerMode() {
        return cpuPowerMode;
    }

    public float inferenceTime() {
        return inferenceTime;
    }
    public Bitmap ReizeByLongandPading(Bitmap inputBitmap,long[] inputShape){
        Bitmap scaledbitmap = Bitmap.createScaledBitmap(inputBitmap, (int) inputShape[3], (int) inputShape[2], true);
        return scaledbitmap;
    }
//    public Bitmap ReizeByLongandPading(Bitmap inputBitmap,long[] inputShape){
//        Mat inputMat=new Mat();
//        int long_size=(int) inputShape[3];
//        pre_shape.clear();
//        double[] paddding_value = {127.5, 127.5, 127.5};
//        org.opencv.android.Utils.bitmapToMat(inputBitmap,inputMat);
//        int origin_w = inputMat.width();
//        int origin_h = inputMat.height();
//        Vector<Integer> origin_resize_shape = new Vector<>();
//        origin_resize_shape.addElement(origin_w);
//        origin_resize_shape.addElement(origin_h);
//        pre_shape.put("resize",origin_resize_shape);
//        int im_size_max = Math.max(origin_w, origin_h);
//        float scale = (float) (long_size) / (float) (im_size_max);
//        int width = Math.round(scale * origin_w);
//        int height = Math.round(scale * origin_h);
//        Size sz = new Size(width, height);
//        Imgproc.resize(inputMat, inputMat, sz,0,0,Imgproc.INTER_LINEAR);
//        Vector<Integer> origin_padding_shape = new Vector<>();
//        origin_padding_shape.addElement(inputMat.width());
//        origin_padding_shape.addElement(inputMat.height());
//        pre_shape.put("pading",origin_padding_shape);
//        //System.out.println("resize:"+inputMat.width()+" "+inputMat.height());
//        double padding_w = long_size-inputMat.width();
//        double padding_h = long_size-inputMat.height();
//        Core.copyMakeBorder(inputMat, inputMat, 0, (int)padding_h, 0, (int)padding_w, Core.BORDER_CONSTANT, new Scalar(paddding_value));
//        //System.out.println("pading:"+inputMat.width()+" "+inputMat.height());
//        Bitmap scaleImage = Bitmap.createBitmap((int) this.config.inputShape[3], (int) this.config.inputShape[2], Bitmap.Config.ARGB_8888);
//        org.opencv.android.Utils.matToBitmap(inputMat,scaleImage);
//        return scaleImage;
//    }
    public void setInputImage(Bitmap image) {
        if (image == null) {
            return;
        }
        // scale image to the size of input tensor
        //System.out.println("运行几次");
        Bitmap rgbaImage = image.copy(Bitmap.Config.ARGB_8888, true);
        Bitmap scaleImage=ReizeByLongandPading(rgbaImage,this.config.inputShape);
        //Bitmap scaleImage = Bitmap.createScaledBitmap(rgbaImage, (int) this.config.inputShape[3], (int) this.config.inputShape[2], true);
        this.inputImage = rgbaImage;
        this.scaledImage = scaleImage;
    }

}
