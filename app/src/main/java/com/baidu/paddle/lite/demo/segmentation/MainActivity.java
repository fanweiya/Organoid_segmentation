package com.baidu.paddle.lite.demo.segmentation;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.ProgressDialog;
import android.content.ContentResolver;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.os.SystemClock;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.demo.segmentation.config.Config;
import com.baidu.paddle.lite.demo.segmentation.preprocess.Preprocess;
import com.baidu.paddle.lite.demo.segmentation.visual.Visualize;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import android.widget.Button;
public class MainActivity extends AppCompatActivity {

    private static final String TAG = MainActivity.class.getSimpleName();
    public static final int OPEN_GALLERY_REQUEST_CODE = 0;
    public static final int TAKE_PHOTO_REQUEST_CODE = 1;

    public static final int REQUEST_LOAD_MODEL = 0;
    public static final int REQUEST_RUN_MODEL = 1;
    public static final int RESPONSE_LOAD_MODEL_SUCCESSED = 0;
    public static final int RESPONSE_LOAD_MODEL_FAILED = 1;
    public static final int RESPONSE_RUN_MODEL_SUCCESSED = 2;
    public static final int RESPONSE_RUN_MODEL_FAILED = 3;

    protected ProgressDialog pbLoadModel = null;
    protected ProgressDialog pbRunModel = null;

    protected Handler receiver = null; // receive messages from worker thread
    protected Handler sender = null; // send command to worker thread
    protected HandlerThread worker = null; // worker thread to load&run model
    protected Button buttonDetectGPU;
    protected Button outputbutton;
    protected Switch vieworiginimg;
    protected Switch CUT_mode;
    protected Switch modelchange;
    protected TextView tvInputSetting;
    protected ImageView ivInputImage;

    protected TextView tvOutputResult;
    protected TextView tvInferenceTime;
    public Bitmap readimage;
    public Bitmap originimage;
    protected EditText bg_thred;
    public int bg_thread_num=0;
    protected EditText area_thred;
    public double area_thread_num=0;
    protected EditText sharp_thred;
    public double sharp_thread_num=1.0;
    protected EditText cut_num_input;
    public double cut_thread_num=2.0;
    protected EditText overlap_num_input;
    public double overlap_thread_num=0.1;
    protected EditText postthrd_num_input;
    public double postthrd_num=0.5;
    protected EditText postiter_num_input;
    public double postiter_num=1;
    public boolean not_cut_detecion=true;
    // model config
    Config config = new Config();

    protected Predictor predictor = new Predictor();

    Preprocess preprocess = new Preprocess();

    Visualize visualize = new Visualize();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        receiver = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case RESPONSE_LOAD_MODEL_SUCCESSED:
                        pbLoadModel.dismiss();
                        onLoadModelSuccessed();
                        break;
                    case RESPONSE_LOAD_MODEL_FAILED:
                        pbLoadModel.dismiss();
                        Toast.makeText(MainActivity.this, "Load model failed!", Toast.LENGTH_SHORT).show();
                        onLoadModelFailed();
                        break;
                    case RESPONSE_RUN_MODEL_SUCCESSED:
                        pbRunModel.dismiss();
                        onRunModelSuccessed();
                        break;
                    case RESPONSE_RUN_MODEL_FAILED:
                        pbRunModel.dismiss();
                        Toast.makeText(MainActivity.this, "Run model failed!", Toast.LENGTH_SHORT).show();
                        onRunModelFailed();
                        break;
                    default:
                        break;
                }
            }
        };

        worker = new HandlerThread("Predictor Worker");
        worker.start();
        sender = new Handler(worker.getLooper()) {
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case REQUEST_LOAD_MODEL:
                        // load model and reload test image
                        if (onLoadModel()) {
                            receiver.sendEmptyMessage(RESPONSE_LOAD_MODEL_SUCCESSED);
                        } else {
                            receiver.sendEmptyMessage(RESPONSE_LOAD_MODEL_FAILED);
                        }
                        break;
                    case REQUEST_RUN_MODEL:
                        // run model if model is loaded
                        if (onRunModel()) {
                            receiver.sendEmptyMessage(RESPONSE_RUN_MODEL_SUCCESSED);
                        } else {
                            receiver.sendEmptyMessage(RESPONSE_RUN_MODEL_FAILED);
                        }
                        break;
                    default:
                        break;
                }
            }
        };
        Utils.initialOpencv();
        tvInputSetting = findViewById(R.id.bg_label);
        ivInputImage = findViewById(R.id.iv_input_image);
        tvInferenceTime = findViewById(R.id.tv_inference_time);
        tvOutputResult = findViewById(R.id.tv_output_result);
        buttonDetectGPU = findViewById(R.id.detect);
        tvInputSetting.setMovementMethod(ScrollingMovementMethod.getInstance());
        tvOutputResult.setMovementMethod(ScrollingMovementMethod.getInstance());
        bg_thred=findViewById(R.id.bg_thred);
        area_thred=findViewById(R.id.area_thred);
        sharp_thred=findViewById(R.id.sharp_thred);
        cut_num_input=findViewById(R.id.cut_num);
        overlap_num_input=findViewById(R.id.overlap_num);
        postthrd_num_input=findViewById(R.id.postthrd_num);
        postiter_num_input=findViewById(R.id.postiter_num);
        outputbutton=findViewById(R.id.outputresult);
        vieworiginimg=findViewById(R.id.vieworigin);
        CUT_mode=findViewById(R.id.CUT_mode);
        cut_num_input.setEnabled(false);
        overlap_num_input.setEnabled(false);
        buttonDetectGPU.setOnClickListener(new View.OnClickListener() {
            @SuppressLint("SetTextI18n")
            @Override
            public void onClick(View arg0) {
                if (readimage != null && predictor.isLoaded()) {
                    bg_thread_num = Integer.parseInt(bg_thred.getText().toString());
                    area_thread_num = Double.parseDouble(area_thred.getText().toString());
                    sharp_thread_num = Double.parseDouble(sharp_thred.getText().toString());
                    cut_thread_num = Double.parseDouble(cut_num_input.getText().toString());
                    overlap_thread_num = Double.parseDouble(overlap_num_input.getText().toString());
                    postthrd_num = Double.parseDouble(postthrd_num_input.getText().toString());
                    postiter_num = Double.parseDouble(postiter_num_input.getText().toString());
                    System.out.println("bg_thread_num "+bg_thread_num);
                    System.out.println("area_thread_num "+area_thread_num);
                    System.out.println("sharp_thread_num "+sharp_thread_num);
                    System.out.println("cut_thread_num "+cut_thread_num);
                    System.out.println("overlap_thread_num "+overlap_thread_num);
                    System.out.println("postthrd_num "+postthrd_num);
                    System.out.println("postiter_num "+postiter_num);
                    predictor.setInputImage(readimage);
                    runModel();
                }

            }
        });
        vieworiginimg.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener(){
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if(isChecked){
                    ivInputImage.setImageBitmap(readimage);
                }else {
                    ivInputImage.setImageBitmap(predictor.outputImage);
                }
            }
        });
        CUT_mode.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener(){
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if(isChecked){
                    not_cut_detecion=false;
                    cut_num_input.setEnabled(true);
                    overlap_num_input.setEnabled(true);
                }else {
                    not_cut_detecion=true;
                    cut_num_input.setEnabled(false);
                    overlap_num_input.setEnabled(false);
                }
            }
        });
        outputbutton.setOnClickListener(new View.OnClickListener() {
            @SuppressLint("SetTextI18n")
            @Override
            public void onClick(View arg0) {
                if (predictor.outputImage != null && predictor.isLoaded()) {
                    predictor.saveBitmap(predictor.filename,predictor.outputImage);
                    Toast.makeText(MainActivity.this, predictor.filename+"????????????-"+predictor.dirPath, Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    public boolean onLoadModel() {
        return predictor.init(MainActivity.this, config);
    }

    public boolean onRunModel() {
        return predictor.isLoaded() && predictor.runModel(preprocess,visualize,bg_thread_num,area_thread_num,sharp_thread_num,cut_thread_num,overlap_thread_num,not_cut_detecion,postthrd_num,postiter_num);
    }

    public void onLoadModelFailed() {

    }
    public void onRunModelFailed() {
    }

    public void loadModel() {
        pbLoadModel = ProgressDialog.show(this, "", "Loading model...", false, false);
        sender.sendEmptyMessage(REQUEST_LOAD_MODEL);
    }

    public void runModel() {
        pbRunModel = ProgressDialog.show(this, "", "Running model...", false, false);
        sender.sendEmptyMessage(REQUEST_RUN_MODEL);
    }

    public void onLoadModelSuccessed() {
        // load test image from file_paths and run model
        try {
            if (config.imagePath.isEmpty()) {
                return;
            }
            Bitmap image = null;
            // read test image file from custom file_paths if the first character of mode file_paths is '/', otherwise read test
            // image file from assets
            if (!config.imagePath.substring(0, 1).equals("/")) {
                InputStream imageStream = getAssets().open(config.imagePath);
                image = BitmapFactory.decodeStream(imageStream);
            } else {
                if (!new File(config.imagePath).exists()) {
                    return;
                }
                image = BitmapFactory.decodeFile(config.imagePath);
            }
            if (image != null && predictor.isLoaded()) {
                readimage=image;
                predictor.setInputImage(image);
                runModel();
            }
        } catch (IOException e) {
            Toast.makeText(MainActivity.this, "Load image failed!", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
    }

    public void onRunModelSuccessed() {
        // obtain results and update UI
        tvInferenceTime.setText("Inference time: " + predictor.inferenceTime() + " ms");
        Bitmap outputImage = predictor.outputImage();
        if (outputImage != null) {
            ivInputImage.setImageBitmap(outputImage);
        }
        tvOutputResult.setText(predictor.outputResult());
        tvOutputResult.scrollTo(0, 0);
    }


    public void onImageChanged(Bitmap image) {
        readimage=image;
        // rerun model if users pick test image from gallery or camera
        if (image != null && predictor.isLoaded()) {
            predictor.setInputImage(image);
            runModel();
        }
    }

    public void onImageChanged(String path) {
        Bitmap image = BitmapFactory.decodeFile(path);
        readimage=image;
        predictor.setInputImage(image);
        runModel();
    }
    public void onSettingsClicked() {
        startActivity(new Intent(MainActivity.this, SettingsActivity.class));
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_action_options, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                finish();
                break;
            case R.id.open_gallery:
                if (requestAllPermissions()) {
                    openGallery();
                }
                break;
            case R.id.take_photo:
                if (requestAllPermissions()) {
                    takePhoto();
                }
                break;
            case R.id.settings:
                if (requestAllPermissions()) {
                    // make sure we have SDCard r&w permissions to load model from SDCard
                    onSettingsClicked();
                }
                break;
        }
        return super.onOptionsItemSelected(item);
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED || grantResults[1] != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
        }
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) {
            switch (requestCode) {
                case OPEN_GALLERY_REQUEST_CODE:
                    try {
                        ContentResolver resolver = getContentResolver();
                        Uri uri = data.getData();
                        Bitmap image = MediaStore.Images.Media.getBitmap(resolver, uri);
                        String[] proj = {MediaStore.Images.Media.DATA};
                        Cursor cursor = managedQuery(uri, proj, null, null, null);
                        cursor.moveToFirst();
                        onImageChanged(image);
                    } catch (IOException e) {
                        Log.e(TAG, e.toString());
                    }
                    break;

                case TAKE_PHOTO_REQUEST_CODE:
                    Bitmap image = (Bitmap) data.getParcelableExtra("data");
                    onImageChanged(image);

                    break;
                default:
                    break;
            }
        }
    }
    private boolean requestAllPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED || ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE,
                            Manifest.permission.CAMERA},
                    0);
            return false;
        }
        return true;
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, null);
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intent, OPEN_GALLERY_REQUEST_CODE);
    }

    private void takePhoto() {
        Intent takePhotoIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePhotoIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePhotoIntent, TAKE_PHOTO_REQUEST_CODE);
        }
    }

    @Override
    public boolean onPrepareOptionsMenu(Menu menu) {
        boolean isLoaded = predictor.isLoaded();
        menu.findItem(R.id.open_gallery).setEnabled(isLoaded);
        menu.findItem(R.id.take_photo).setEnabled(isLoaded);
        return super.onPrepareOptionsMenu(menu);
    }

    @Override
    protected void onResume() {
        Log.i(TAG,"begin onResume");
        super.onResume();

        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        boolean settingsChanged = false;
        String model_path = sharedPreferences.getString(getString(R.string.MODEL_PATH_KEY),
                getString(R.string.MODEL_PATH_DEFAULT));
        String label_path = sharedPreferences.getString(getString(R.string.LABEL_PATH_KEY),
                getString(R.string.LABEL_PATH_DEFAULT));
        String image_path = sharedPreferences.getString(getString(R.string.IMAGE_PATH_KEY),
                getString(R.string.IMAGE_PATH_DEFAULT));
        settingsChanged |= !model_path.equalsIgnoreCase(config.modelPath);
        settingsChanged |= !label_path.equalsIgnoreCase(config.labelPath);
        settingsChanged |= !image_path.equalsIgnoreCase(config.imagePath);
        int cpu_thread_num = Integer.parseInt(sharedPreferences.getString(getString(R.string.CPU_THREAD_NUM_KEY),
                getString(R.string.CPU_THREAD_NUM_DEFAULT)));
        settingsChanged |= cpu_thread_num != config.cpuThreadNum;
        String cpu_power_mode =
                sharedPreferences.getString(getString(R.string.CPU_POWER_MODE_KEY),
                        getString(R.string.CPU_POWER_MODE_DEFAULT));
        settingsChanged |= !cpu_power_mode.equalsIgnoreCase(config.cpuPowerMode);
        String input_color_format =
                sharedPreferences.getString(getString(R.string.INPUT_COLOR_FORMAT_KEY),
                        getString(R.string.INPUT_COLOR_FORMAT_DEFAULT));
        settingsChanged |= !input_color_format.equalsIgnoreCase(config.inputColorFormat);
        long[] input_shape =
                Utils.parseLongsFromString(sharedPreferences.getString(getString(R.string.INPUT_SHAPE_KEY),
                        getString(R.string.INPUT_SHAPE_DEFAULT)), ",");

        settingsChanged |= input_shape.length != config.inputShape.length;

        if (!settingsChanged) {
            for (int i = 0; i < input_shape.length; i++) {
                settingsChanged |= input_shape[i] != config.inputShape[i];
            }
        }

        if (settingsChanged) {
            config.init(model_path,label_path,image_path,cpu_thread_num,cpu_power_mode,
                    input_color_format,input_shape);
            preprocess.init(config);
            // update UI
//            tvInputSetting.setText("Model: " + config.modelPath.substring(config.modelPath.lastIndexOf("/") + 1) + "\n" + "CPU" +
//                    " Thread Num: " + Integer.toString(config.cpuThreadNum) + "\n" + "CPU Power Mode: " + config.cpuPowerMode);
//            tvInputSetting.scrollTo(0, 0);
            // reload model if configure has been changed
            loadModel();
        }
    }

    @Override
    protected void onDestroy() {
        if (predictor != null) {
            predictor.releaseModel();
        }
        worker.quit();
        super.onDestroy();
    }
}
