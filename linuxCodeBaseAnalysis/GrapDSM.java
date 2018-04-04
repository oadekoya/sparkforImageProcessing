import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;

@SuppressWarnings("serial")

class DSMPanel extends JPanel {

	static double[][] matrix;
	static String fileName;
	static int ovalSize;
	static int upSet;
	static int statingPoint;

	@SuppressWarnings("resource")
	private void doDrawing(Graphics g) {

		Graphics2D graphics = (Graphics2D) g;

		graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		graphics.setPaint(Color.blue);

		String line;

		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));

			int numberOfLinesInFile = 0;

			while ((line = reader.readLine()) != null) {
				numberOfLinesInFile++;
			}

			matrix = new double[numberOfLinesInFile][numberOfLinesInFile];

			reader = new BufferedReader(new FileReader(fileName));

			int lineCount = 0;

			while ((line = reader.readLine()) != null) {
				String[] numbers = line.split(",");
				for (int i = 0; i < matrix.length; i++) {
					matrix[lineCount][i] = Double.parseDouble(numbers[i]);
				}
				lineCount++;
			}

			reader.close();

		} catch (Exception ex) {
			ex.printStackTrace();
		}

		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix.length; j++) {
				if (matrix[i][j] > 0) {
					int x = (int) i + statingPoint;
					int y = (int) j + statingPoint;
					graphics.fillOval(x * upSet, y * upSet, ovalSize, ovalSize);
				}
			}
		}
	}

	@Override
	public void paintComponent(Graphics g) {

		super.paintComponent(g);
		doDrawing(g);
	}
}

public class GrapDSM {

	static int PanelSize;
	static int frameWidth;
	static int frameHeight;
	static boolean toMaximize = false;
	static String imageName;

	public static void main(String[] args) {

		if (args.length < 3) {
			System.out.println("");
			System.out.println(
					"Usage: java GrapDSM [input_file_path_and_name] [output_file_path] [dsm_modularity_level]");
		} else {

			if (args[2].equals("file")) {

				DSMPanel.fileName = args[0];
				DSMPanel.ovalSize = 5;
				DSMPanel.upSet = 1;
				DSMPanel.statingPoint = 20;

				PanelSize = 2150;
				frameWidth = 1500;
				frameHeight = 1000;

				toMaximize = true;
				imageName = "fileDsmImage.png";

			} else if (args[2].equals("group")) {

				DSMPanel.fileName = args[0];

				DSMPanel.ovalSize = 10;
				DSMPanel.upSet = 10;
				DSMPanel.statingPoint = 10;

				PanelSize = 2000;
				frameWidth = 900;
				frameHeight = 900;

				imageName = "groupDsmImage.png";

			} else {
				System.out.println("Modularity level invalid");
				System.exit(1);
			}

			JFrame frame = new JFrame();

			JPanel panel = new DSMPanel();
			panel.setPreferredSize(new Dimension(PanelSize, PanelSize));

			frame.setLayout(new BorderLayout());

			JScrollPane scrPane = new JScrollPane(panel);

			frame.add(scrPane, BorderLayout.CENTER);

			frame.setSize(frameWidth, frameHeight);
			frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			frame.setLocationRelativeTo(null);
			if (toMaximize) {
				frame.setExtendedState(JFrame.MAXIMIZED_BOTH);
			}
			frame.setVisible(true);

			BufferedImage bi = new BufferedImage(panel.getWidth(), panel.getHeight(), BufferedImage.TYPE_INT_ARGB);
			Graphics g = bi.createGraphics();
			panel.paint(g);
			g.dispose();
			try {
				ImageIO.write(bi, "png", new File(args[1] + imageName));
			} catch (Exception e) {
				e.printStackTrace();
			}

			frame.dispose();
		}
		System.exit(0);
	}
}