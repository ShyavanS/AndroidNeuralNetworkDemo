package com.example.androidneuralnetworkdemo;

import android.Manifest;
import android.app.AlertDialog;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {
    private final Path home = Paths.get(System.getenv("HOME"));

    private final String[] permissions = new String[] { Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE };

    private final Python py = Python.getInstance();

    private PyObject backend;

    private Button randomButton;
    private Button saveButton;
    private Button scatterButton;
    private Button lossButton;
    private EditText epochs;
    private EditText rate;

    private final ActivityResultLauncher<String[]> requestPermissionLauncher = registerForActivityResult(
            new ActivityResultContracts.RequestMultiplePermissions(), isGranted -> {
                if (Boolean.TRUE.equals(isGranted.get("android.permission.READ_EXTERNAL_STORAGE"))
                        && Boolean.TRUE.equals(isGranted.get("android.permission.WRITE_EXTERNAL_STORAGE"))) {
                    backend = py.getModule("backend");
                } else {
                    Toast.makeText(this, "You need to provide read/write permission for storage of the json file.",
                            Toast.LENGTH_LONG).show();

                    this.finishAffinity();
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
                && ContextCompat.checkSelfPermission(this,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
            backend = py.getModule("backend");
        } else {
            requestPermissionLauncher.launch(permissions);
        }

        randomButton = findViewById(R.id.randomButton);
        saveButton = findViewById(R.id.saveButton);
        scatterButton = findViewById(R.id.scatterButton);
        lossButton = findViewById(R.id.lossButton);
        epochs = findViewById(R.id.editTextNumber);
        rate = findViewById(R.id.editTextNumberDecimal);

        saveButton.setEnabled(false);
        scatterButton.setEnabled(false);
        lossButton.setEnabled(false);
    }

    public void onClickTrain(View view) {
        int epochNum = Integer.parseInt(epochs.getText().toString());
        float rateNum = Float.parseFloat(rate.getText().toString());

        double rms = Float.parseFloat(backend.callAttr("train_model", epochNum, rateNum).toString());

        AlertDialog.Builder trained = new AlertDialog.Builder(MainActivity.this);

        trained.setTitle("Model Trained!");
        trained.setMessage(String.format(Locale.getDefault(), "The RMS error value of this model is: %f.", rms));
        trained.setPositiveButton("Dismiss", (dialog, id) -> dialog.dismiss());

        AlertDialog trainedDialog = trained.create();

        trainedDialog.show();

        saveButton.setEnabled(true);
        scatterButton.setEnabled(true);
        lossButton.setEnabled(true);
    }

    public void onClickRandom(View view) {
        boolean data_holdover_empty = Boolean.parseBoolean(backend.callAttr("random_select").toString());

        AlertDialog.Builder selected = new AlertDialog.Builder(MainActivity.this);

        selected.setTitle("Random Datapoint Added!");
        selected.setMessage(
                "A random datapoint has been chosen to be added to the dataset, press the Train button to retrain the model with new data.");
        selected.setPositiveButton("Dismiss", (dialog, id) -> dialog.dismiss());

        AlertDialog trainedDialog = selected.create();

        trainedDialog.show();

        if (data_holdover_empty) {
            randomButton.setEnabled(false);
        }
    }

    public void onClickLoad(View view) {
        final Spinner models = new Spinner(MainActivity.this);

        File loadDir = new File(home.toString());

        File[] files = loadDir.listFiles((dir, name) -> name.endsWith(".json"));

        String[] filenames = new String[0];
        if (files != null) {
            filenames = new String[files.length];

            for (int i = 0; i < files.length; i++) {
                filenames[i] = files[i].getName().replaceAll(".json", "");
            }
        }

        ArrayAdapter<String> modelsAdapter = new ArrayAdapter<>(MainActivity.this, android.R.layout.simple_spinner_item,
                filenames);

        modelsAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        models.setAdapter(modelsAdapter);

        AlertDialog.Builder load = new AlertDialog.Builder(MainActivity.this);

        load.setTitle("Load Model");
        load.setMessage("Which model would you like to load?");
        load.setView(models);
        load.setNegativeButton("Cancel", (dialog, id) -> dialog.cancel());
        load.setPositiveButton("Load", (dialog, id) -> {
            backend.callAttr("load", models.getSelectedItem().toString());

            saveButton.setEnabled(true);
            scatterButton.setEnabled(true);
            lossButton.setEnabled(true);
        });

        AlertDialog saveDialog = load.create();

        saveDialog.show();
    }

    public void onClickSave(View view) {
        final EditText saveText = new EditText(MainActivity.this);

        AlertDialog.Builder save = new AlertDialog.Builder(MainActivity.this);

        save.setTitle("Save Model");
        save.setMessage("What would you like to call the model?");
        save.setView(saveText);
        save.setPositiveButton("Save", (dialog, id) -> backend.callAttr("save", saveText.getText().toString()));
        save.setNegativeButton("Cancel", (dialog, id) -> dialog.cancel());

        AlertDialog saveDialog = save.create();

        saveDialog.show();
    }

    public void onClickScatter(View view) {
        final ImageView scatterPlot = new ImageView(MainActivity.this);

        backend.callAttr("plot_scatter");

        Path scatterImg = Paths.get(home.toString(), "scatter.png");
        Bitmap scatterBitmap = BitmapFactory.decodeFile(scatterImg.toString());

        scatterPlot.setImageBitmap(scatterBitmap);

        AlertDialog.Builder scatter = new AlertDialog.Builder(MainActivity.this);

        scatter.setTitle("Scatter Plot Made!");
        scatter.setMessage(String.format(Locale.getDefault(), "Path to image: %s", scatterImg));
        scatter.setView(scatterPlot);
        scatter.setPositiveButton("Dismiss", (dialog, id) -> dialog.dismiss());

        AlertDialog scatterDialog = scatter.create();

        scatterDialog.show();
    }

    public void onClickLoss(View view) {
        final ImageView lossPlot = new ImageView(MainActivity.this);

        backend.callAttr("plot_loss");

        Path lossImg = Paths.get(home.toString(), "loss.png");
        Bitmap lossBitmap = BitmapFactory.decodeFile(lossImg.toString());

        lossPlot.setImageBitmap(lossBitmap);

        AlertDialog.Builder loss = new AlertDialog.Builder(MainActivity.this);

        loss.setTitle("Scatter Plot Made!");
        loss.setMessage(String.format(Locale.getDefault(), "Path to image: %s", lossImg));
        loss.setView(lossPlot);
        loss.setPositiveButton("Dismiss", (dialog, id) -> dialog.dismiss());

        AlertDialog lossDialog = loss.create();

        lossDialog.show();
    }
}
