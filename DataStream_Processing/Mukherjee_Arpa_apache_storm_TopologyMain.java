import backtype.storm.Config;
import backtype.storm.LocalCluster;
import backtype.storm.topology.TopologyBuilder;
import backtype.storm.tuple.Fields;
import backtype.storm.utils.Utils;
import spouts.WordReader;
import bolts.WordCounter;
import bolts.WordNormalizer;


public class TopologyMain {
	public static void main(String[] args) throws InterruptedException {
     
	    TopologyBuilder builder = new TopologyBuilder();

	    builder.setSpout("word", new WordReader(), 1);
	    builder.setBolt("normalize", new WordNormalizer(), 3).shuffleGrouping("word");
	    builder.setBolt("count", new WordCounter(), 1).shuffleGrouping("normalize");

	    Config conf = new Config();
	    conf.setDebug(true);
	    conf.put("wordsFile", "./resources/words.txt");

	    LocalCluster cluster = new LocalCluster();
	    cluster.submitTopology("test", conf, builder.createTopology());
	    Utils.sleep(10000);
	    cluster.killTopology("test");
	    cluster.shutdown();
		
		
		
	}
}

