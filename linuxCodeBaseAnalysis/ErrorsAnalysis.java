import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;

public class ErrorsAnalysis {

	static String str;
	static List<String> fileContent = new ArrayList<String>();
	static List<String> errorStatistics = new ArrayList<String>();
	static TreeMap<String, String> mergedErrorStatistics = new TreeMap<String, String>();
	static List<String> treeMapToList = new ArrayList<String>();
	static List<String> finalReport = new ArrayList<String>();

	static String[] arrayFileContent;

	static int totalErrors = 0;
	static int totalWarnings = 0;
	static int totalNotes = 0;

	public static void main(String[] args) {

		if (args.length < 2) {
			System.out.println("");
			System.out.println("Usage: java ErrorsAnalysis [input_file_path_and_name] [output_file_path]");
		} else {

			try {

				BufferedReader reader = new BufferedReader(new FileReader(args[0]));

				while ((str = reader.readLine()) != null) {
					fileContent.add(str);
				}
				reader.close();
			} catch (Exception ex) {
				ex.printStackTrace();
			}

			arrayFileContent = fileContent.toArray(new String[0]);

			int counter = 0;

			String previousFileName = "";
			String previousErrorName = "";
			String errorName = "";
			String fileName = "";

			for (int i = 0; i < arrayFileContent.length; i++) {

				String[] splitLine = arrayFileContent[i].split(":");

				if (splitLine.length > 3) {

					fileName = splitLine[0].trim();
					errorName = splitLine[3].trim();

					if (previousFileName.equals("")) {
						previousFileName = fileName;
					}

					if (previousErrorName.equals("")) {
						previousErrorName = errorName;
					}

					if (previousFileName.equals(fileName) && errorName.equals(previousErrorName)) {
						counter++;
					} else {
						errorStatistics.add(previousFileName + "," + previousErrorName + "," + counter);
						counter = 1;
					}
				}

				previousFileName = fileName;
				previousErrorName = errorName;
			}

			errorStatistics.add(previousFileName + "," + errorName + "," + counter);

			String key_1 = "";
			String key_2 = "";
			int totalValue = 0;

			for (int i = 0; i < errorStatistics.size(); i++) {

				String[] splitLine_1 = errorStatistics.get(i).split(",");
				key_1 = splitLine_1[0] + "," + splitLine_1[1] + ",";

				if (!mergedErrorStatistics.containsKey(key_1)) {
					for (int j = 0; j < errorStatistics.size(); j++) {
						String[] splitLine_2 = errorStatistics.get(j).split(",");
						key_2 = splitLine_2[0] + "," + splitLine_2[1] + ",";
						int value_2 = 0;
						try {
							value_2 = Integer.valueOf(splitLine_2[2]);
						} catch (Exception ex) {
							continue;
						}

						if (key_1.equals(key_2)) {
							totalValue += value_2;
						}
					}

					mergedErrorStatistics.put(key_1, Integer.toString(totalValue));
				}

				totalValue = 0;
			}

			for (String key : mergedErrorStatistics.keySet()) {
				String value = mergedErrorStatistics.get(key);
				treeMapToList.add(key + value);
			}

			finalReport.add("fileName,errorsCount,warningsCount,notesCount");

			for (int i = 0; i < treeMapToList.size(); i++) {

				int errors_1 = 0;
				int warning_1 = 0;
				int notes_1 = 0;

				int errors_2 = 0;
				int warning_2 = 0;
				int notes_2 = 0;

				int errorsCount = 0;
				int warningCount = 0;
				int notesCount = 0;

				String[] splitLine_1 = treeMapToList.get(i).split(",");
				String prvFileName = splitLine_1[0];

				if (splitLine_1[1].equals("error")) {
					errors_1 = Integer.valueOf(splitLine_1[2]);
				} else if (splitLine_1[1].equals("warning")) {
					warning_1 = Integer.valueOf(splitLine_1[2]);
				} else if (splitLine_1[1].equals("note")) {
					notes_1 = Integer.valueOf(splitLine_1[2]);
				}

				for (int j = 0; j < treeMapToList.size(); j++) {

					String[] splitLine_2 = treeMapToList.get(j).split(",");
					String currentFileName = splitLine_2[0];

					if (prvFileName.equals(currentFileName)) {

						if (splitLine_2[1].equals("error")) {
							errors_2 = Integer.valueOf(splitLine_2[2]);
						} else if (splitLine_2[1].equals("warning")) {
							warning_2 = Integer.valueOf(splitLine_2[2]);
						} else if (splitLine_2[1].equals("note")) {
							notes_2 = Integer.valueOf(splitLine_2[2]);
						}
					}
				}

				errorsCount = errors_1 == 0 ? errors_2 : errors_1;
				warningCount = warning_1 == 0 ? warning_2 : warning_1;
				notesCount = notes_1 == 0 ? notes_2 : notes_1;

				if (!(finalReport.contains(prvFileName + "," + errorsCount + "," + warningCount + "," + notesCount))) {
					finalReport.add(prvFileName + "," + errorsCount + "," + warningCount + "," + notesCount);
				}
			}

			for (int i = 1; i < finalReport.size(); i++) {
				String[] splitLine = finalReport.get(i).split(",");
				totalErrors += Integer.valueOf(splitLine[1]);
				totalWarnings += Integer.valueOf(splitLine[2]);
				totalNotes += Integer.valueOf(splitLine[3]);
			}

			finalReport.add(" ");
			finalReport.add("Total errors = " + totalErrors + ", Total warnings = " + totalWarnings + ", Total notes = "
					+ totalNotes);

			try {

				FileWriter writer = new FileWriter(args[1] + "/errors_statistics.txt");

				for (String line : finalReport) {
					writer.write(line + "\n");
				}

				writer.close();
				System.out.println("");
				System.out.println("Done...");
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
}
