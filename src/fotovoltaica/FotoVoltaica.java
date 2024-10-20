package fotovoltaica;

import java.awt.Desktop;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.InputStream;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileNameExtensionFilter;
import view.Home;

public class FotoVoltaica {
    public static void main(String[] args) {
        Home h = new Home();
        h.setVisible(true);
    }
}