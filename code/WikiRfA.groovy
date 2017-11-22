package edu.umd.cs.bachuai13

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import com.google.common.collect.Iterables

import edu.umd.cs.bachuai13.util.DataOutputter;
import edu.umd.cs.bachuai13.util.ExperimentConfigGenerator;
import edu.umd.cs.bachuai13.util.FoldUtils;
import edu.umd.cs.bachuai13.util.GroundingWrapper;
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.VotedPerceptron
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.LossBalancingType
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.NormScalingType
import edu.umd.cs.psl.application.learning.weight.random.FirstOrderMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.GroundMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.IncompatibilityMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.MetropolisRandOM
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.core.*
import edu.umd.cs.psl.core.inference.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.*
import edu.umd.cs.psl.evaluation.statistics.RankingScore
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.Model
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.atom.GroundAtom
import edu.umd.cs.psl.model.atom.RandomVariableAtom
import edu.umd.cs.psl.model.kernel.CompatibilityKernel
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.ui.loading.*
import edu.umd.cs.psl.util.database.Queries


ExperimentConfigGenerator configGenerator = new ExperimentConfigGenerator("wiki_rfa");

def procId = args[0];
def modelType = args[1]
def numTopFeatsToDiscard = args[2]
String obsRatio = args[3]
String evalType = args[4]
boolean joint = (args[5] == "joint")
def config_train = args[6]
def config_test = args[7]
boolean noObservationsForTesting = args[8].equals("1")

def posCutoff = -1
def negCutoff = -1
if (evalType.equals("lexiconBased")) {
	posCutoff = args[9]
	negCutoff = args[10]
}


/*
* SET MODEL TYPES
*
* Options:
* "quad" HL-MRF-Q
* "linear" HL-MRF-L
* "bool" MRF
*/
configGenerator.setModelTypes([modelType]);

/*
 * SET LEARNING ALGORITHMS
 * 
 * Options:
 * "MLE" (MaxLikelihoodMPE)
 * "MPLE" (MaxPseudoLikelihood)
 * "MM" (MaxMargin)
 */
configGenerator.setLearningMethods(["MLE"]);
//configGenerator.setLearningMethods(["MLE", "MPLE", "MM"]);

/* MLE/MPLE options */
configGenerator.setVotedPerceptronStepCounts([100]);
configGenerator.setVotedPerceptronStepSizes([(double) 5.0]);

/* MM options */
configGenerator.setMaxMarginSlackPenalties([(double) 0.1]);
configGenerator.setMaxMarginLossBalancingTypes([LossBalancingType.NONE]);
configGenerator.setMaxMarginNormScalingTypes([NormScalingType.NONE]);
configGenerator.setMaxMarginSquaredSlackValues([false]);

Logger log = LoggerFactory.getLogger(this.class)

boolean sq = (!modelType.equals("linear") ? true : false)

List<ConfigBundle> configs = configGenerator.getConfigs();

/*
 * PRINTS EXPERIMENT CONFIGURATIONS
 */
for (ConfigBundle config : configs)
	System.out.println(config);

/*
 * INITIALIZES DATASTORE AND MODEL
 */
ConfigManager cm = ConfigManager.getManager();
ConfigBundle baseConfig = cm.getBundle("wiki_rfa");
def defaultPath = System.getProperty("java.io.tmpdir")
// We need to add the procId to the path, so we can run several processes in parallel (otherwise
// the DB will be locked.)
String dbpath = baseConfig.getString("dbpath", defaultPath + "/psl-wiki_rfa/" + evalType + "/" + procId)
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), baseConfig)
PSLModel m = new PSLModel(this, data)

/*
 * DEFINE MODEL
 */
m.add predicate: "knows", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "trusts", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "prior", types: [ArgumentType.UniqueID]
m.add predicate: "logreg", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
// To allow for different deviations for different values of LR prediction; args: [node1, node2, logregBin].
m.add predicate: "logreg_bin", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]

double initialWeight = 1

// Acyclical.
m.add rule: (knows(A,B) & knows(B,C) & knows(A,C) & (A - B) & (B - C) & (A - C)) >> ~trusts(A,B) | ~trusts(B,C) | ~trusts(A,C), weight: initialWeight, squared: sq
m.add rule: (knows(A,B) & knows(B,C) & knows(A,C) & (A - B) & (B - C) & (A - C)) >> ~trusts(A,B) | ~trusts(B,C) |  trusts(A,C), weight: initialWeight, squared: sq
m.add rule: (knows(A,B) & knows(B,C) & knows(A,C) & (A - B) & (B - C) & (A - C)) >> ~trusts(A,B) |  trusts(B,C) | ~trusts(A,C), weight: initialWeight, squared: sq
m.add rule: (knows(A,B) & knows(B,C) & knows(A,C) & (A - B) & (B - C) & (A - C)) >> ~trusts(A,B) |  trusts(B,C) |  trusts(A,C), weight: initialWeight, squared: sq
m.add rule: (knows(A,B) & knows(B,C) & knows(A,C) & (A - B) & (B - C) & (A - C)) >>  trusts(A,B) | ~trusts(B,C) | ~trusts(A,C), weight: initialWeight, squared: sq
m.add rule: (knows(A,B) & knows(B,C) & knows(A,C) & (A - B) & (B - C) & (A - C)) >>  trusts(A,B) | ~trusts(B,C) |  trusts(A,C), weight: initialWeight, squared: sq
m.add rule: (knows(A,B) & knows(B,C) & knows(A,C) & (A - B) & (B - C) & (A - C)) >>  trusts(A,B) |  trusts(B,C) | ~trusts(A,C), weight: initialWeight, squared: sq
m.add rule: (knows(A,B) & knows(B,C) & knows(A,C) & (A - B) & (B - C) & (A - C)) >>  trusts(A,B) |  trusts(B,C) |  trusts(A,C), weight: initialWeight, squared: sq
// Cyclical.
m.add rule: (knows(A,B) & knows(B,C) & knows(C,A) & (A - B) & (B - C) & (A - C)) >>  ~trusts(A,B) | ~trusts(B,C) | ~trusts(C,A), weight: initialWeight, squared: sq
m.add rule: (knows(A,B) & knows(B,C) & knows(C,A) & (A - B) & (B - C) & (A - C)) >>  ~trusts(A,B) | ~trusts(B,C) |  trusts(C,A), weight: initialWeight, squared: sq
m.add rule: (knows(A,B) & knows(B,C) & knows(C,A) & (A - B) & (B - C) & (A - C)) >>  ~trusts(A,B) |  trusts(B,C) |  trusts(C,A), weight: initialWeight, squared: sq
m.add rule: (knows(A,B) & knows(B,C) & knows(C,A) & (A - B) & (B - C) & (A - C)) >>   trusts(A,B) |  trusts(B,C) |  trusts(C,A), weight: initialWeight, squared: sq
// Reciprocation.
//m.add rule: (knows(A,B) & knows(B,A) & trusts(A,B)) >>  trusts(B,A), weight: initialWeight, squared: sq
//m.add rule: (knows(A,B) & knows(B,A) & trusts(A,B)) >> ~trusts(B,A), weight: initialWeight, squared: sq

// two-sided prior
UniqueID constant = data.getUniqueID(0)
m.add rule: (knows(A,B) & prior(constant)) >> trusts(A,B), weight: initialWeight, squared: sq
m.add rule: (knows(A,B) & trusts(A,B)) >> prior(constant), weight: initialWeight, squared: sq

if (joint) {
	// Logistic regression predictions.
	//m.add rule: (knows(A,B) & logreg(A,B)) >> trusts(A,B), weight: initialWeight, squared: false
	//m.add rule: (knows(A,B) & trusts(A,B)) >> logreg(A,B), weight: initialWeight, squared: false
	// More fine-grained version.
	for (int i = 0; i <= 9; ++i) {
		def bin = data.getUniqueID(i)
		m.add rule: (knows(A,B) & logreg_bin(A,B,bin) & logreg(A,B)) >> trusts(A,B), weight: initialWeight, squared: false
		m.add rule: (knows(A,B) & logreg_bin(A,B,bin) & trusts(A,B)) >> logreg(A,B), weight: initialWeight, squared: false
	}
}

// save all initial weights
Map<CompatibilityKernel,Weight> weights = new HashMap<CompatibilityKernel, Weight>()
for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
	weights.put(k, k.getWeight());

/*
 * LOAD DATA
 */
def BASEDIR = System.getenv('HOME') + "/repo/sentistruct/data/wiki_rfa/optimization-data/"
def dataPaths = [BASEDIR+config_train+"/", BASEDIR+config_test+"/"]
int folds = 1
// int folds = dataPaths.size()

List<List<Double []>> results = new ArrayList<List<Double []>>()
for (int i = 0; i < configs.size(); i++)
	results.add(new ArrayList<Double []>())

for (int fold = 0; fold < folds; fold++) {

	Partition read_tr   = new Partition(fold + 0 * folds)
	Partition write_tr  = new Partition(fold + 1 * folds)
	Partition read_te   = new Partition(fold + 2 * folds)
	Partition write_te  = new Partition(fold + 3 * folds)
	Partition labels_tr = new Partition(fold + 4 * folds)
	Partition labels_te = new Partition(fold + 5 * folds)

	def trainPath = dataPaths[fold]
	def testPath = dataPaths[fold+1]
	//def testPath = dataPaths[(fold+1)%folds]

	//System.err.format("trainPath: %s\n", trainPath)
	//System.err.format("testPath: %s\n", testPath)

	// Insert data into partitions.
	def inserter
	// voted
	inserter = data.getInserter(knows, read_tr);
	InserterUtils.loadDelimitedDataTruth(inserter, trainPath + "voted.txt");
	inserter = data.getInserter(knows, read_te);
	InserterUtils.loadDelimitedDataTruth(inserter, testPath + "voted.txt");
	if (evalType.equals("varyingObsRatio") || evalType.equals("varyingDiscardedFeats")) {
		// supported
		inserter = data.getInserter(trusts, read_tr);
		InserterUtils.loadDelimitedDataTruth(inserter, trainPath + "supported_obsRatio=" + obsRatio + "_obs.txt");
		inserter = data.getInserter(trusts, read_te);
		if (noObservationsForTesting) {
			// Nothing to be inserted as observed for testing.
		} else {
			InserterUtils.loadDelimitedDataTruth(inserter, testPath + "supported_obsRatio=" + obsRatio + "_obs.txt");
		}
		// labels
		inserter = data.getInserter(trusts, labels_tr);
		InserterUtils.loadDelimitedDataTruth(inserter, trainPath + "supported_obsRatio=" + obsRatio + "_hid.txt");
		inserter = data.getInserter(trusts, labels_te);
		if (noObservationsForTesting) {
			// Everything is hidden if nothing is observed during testing.
			InserterUtils.loadDelimitedDataTruth(inserter, testPath + "supported.txt");
		} else {
			InserterUtils.loadDelimitedDataTruth(inserter, testPath + "supported_obsRatio=" + obsRatio + "_hid.txt");
		}
		// logreg
		inserter = data.getInserter(logreg, read_tr);
		InserterUtils.loadDelimitedDataTruth(inserter, trainPath + "logreg_discard-top-" + numTopFeatsToDiscard + "-feats.txt");
		inserter = data.getInserter(logreg, read_te);
		InserterUtils.loadDelimitedDataTruth(inserter, testPath + "logreg_discard-top-" + numTopFeatsToDiscard + "-feats.txt");
	} else if (evalType.equals("lexiconBased")) {
		// supported
		inserter = data.getInserter(trusts, read_tr);
		InserterUtils.loadDelimitedDataTruth(inserter, trainPath + "supported_lexiconBased_posCutoff=" + posCutoff + "_negCutoff=" + negCutoff + "_obs.txt");
		inserter = data.getInserter(trusts, read_te);
		InserterUtils.loadDelimitedDataTruth(inserter, testPath + "supported_lexiconBased_posCutoff=" + posCutoff + "_negCutoff=" + negCutoff + "_obs.txt");
		// labels
		inserter = data.getInserter(trusts, labels_tr);
		InserterUtils.loadDelimitedDataTruth(inserter, trainPath + "supported_lexiconBased_posCutoff=" + posCutoff + "_negCutoff=" + negCutoff + "_hid.txt");
		inserter = data.getInserter(trusts, labels_te);
		InserterUtils.loadDelimitedDataTruth(inserter, testPath + "supported_lexiconBased_posCutoff=" + posCutoff + "_negCutoff=" + negCutoff + "_hid.txt");
		// logreg (NB: we're abusing the 'logreg' label here, since the values really come from the lexicon).
		inserter = data.getInserter(logreg, read_tr);
		InserterUtils.loadDelimitedDataTruth(inserter, trainPath + "lexicon_probs.txt");
		inserter = data.getInserter(logreg, read_te);
		InserterUtils.loadDelimitedDataTruth(inserter, testPath + "lexicon_probs.txt");
	} else {
		throw new IllegalArgumentException(evalType + " is not a legal evalType value");
	}


	// Add the edges to their respective logreg bins.
	// training
	Database trainDB = data.getDatabase(write_tr, read_tr);
	Set<GroundAtom> allLogreg = Queries.getAllAtoms(trainDB, logreg)
	trainDB.close()
	inserter = data.getInserter(logreg_bin, read_tr);
	for (GroundAtom atom : allLogreg) {
		GroundTerm[] atomArgs = atom.getArguments()
		GroundTerm[] a = [ atomArgs[0], atomArgs[1], data.getUniqueID((int)(10*atom.getValue())) ]
		inserter.insertValue(1.0, a)
	}
	// testing
	Database testDB = data.getDatabase(write_te, read_te);
	allLogreg = Queries.getAllAtoms(testDB, logreg)
	testDB.close()
	inserter = data.getInserter(logreg_bin, read_te);
	for (GroundAtom atom : allLogreg) {
		GroundTerm[] atomArgs = atom.getArguments()
		GroundTerm[] a = [ atomArgs[0], atomArgs[1], data.getUniqueID((int)(10*atom.getValue())) ]
		inserter.insertValue(1.0, a)
	}

	// Compute priors.
	// training
	trainDB = data.getDatabase(write_tr, read_tr);
	Set<GroundAtom> allTrusts = Queries.getAllAtoms(trainDB, trusts)
	double sum = 0.0;
	for (GroundAtom atom : allTrusts)
		sum += atom.getValue()
	trainDB.close()
	data.getInserter(prior, read_tr).insertValue(sum / allTrusts.size(), constant)
	log.info("Computed training prior for fold {} of {}", fold, sum / allTrusts.size())
	// testing
	testDB = data.getDatabase(write_te, read_te);
	allTrusts = Queries.getAllAtoms(testDB, trusts)
	sum = 0.0;
	for (GroundAtom atom : allTrusts)
		sum += atom.getValue()
	testDB.close()
	data.getInserter(prior, read_te).insertValue(sum / allTrusts.size(), constant)
	log.info("Computed testing prior for fold {} of {}", fold, sum / allTrusts.size())

	// reopen databases
	toClose = [knows, prior, logreg] as Set
	trainDB = data.getDatabase(write_tr, toClose, read_tr)
	testDB = data.getDatabase(write_te, toClose, read_te)

	/*
	 * POPULATE TRAINING DATABASE
	 * Get all knows pairs, 
	 */
	int rv = 0, ob = 0
	ResultList allGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(knows))
	for (int i = 0; i < allGroundings.size(); i++) {
		GroundTerm [] grounding = allGroundings.get(i)
		GroundAtom atom = trainDB.getAtom(trusts, grounding)
		if (atom instanceof RandomVariableAtom) {
			rv++
			trainDB.commit((RandomVariableAtom) atom);
		} else {
			ob++
		}
	}
	System.out.println("Saw " + rv + " rvs and " + ob + " obs")

	/*
	 * POPULATE TEST DATABASE
	 */
	allGroundings = testDB.executeQuery(Queries.getQueryForAllAtoms(knows))
	for (int i = 0; i < allGroundings.size(); i++) {
		GroundTerm [] grounding = allGroundings.get(i)
		GroundAtom atom = testDB.getAtom(trusts, grounding)
		if (atom instanceof RandomVariableAtom) {
			testDB.commit((RandomVariableAtom) atom);
		}
	}

	testDB.close()

	Partition dummy = new Partition(99999)
	Partition dummy2 = new Partition(19999)
	Database labelsDB = data.getDatabase(dummy, [trusts] as Set, labels_tr)
	
	DataOutputter.outputPredicate("output/wiki_rfa/training-truth" + fold + ".directed" , labelsDB, trusts, ",", true, "Source,Target,TrueTrusts");

	for (int configIndex = 0; configIndex < configs.size(); configIndex++) {
		ConfigBundle config = configs.get(configIndex);
		for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
			k.setWeight(weights.get(k))

		/*
		 * Weight learning
		 */
		learn(m, trainDB, labelsDB, config, log)

		System.out.println("Learned model " + config.getString("name", "") + "\n" + m.toString())

		/*
		 * Inference on test set
		 */
		testDB = data.getDatabase(write_te, read_te)
		for (int i = 0; i < allGroundings.size(); i++) {
			GroundTerm [] grounding = allGroundings.get(i)
			GroundAtom atom = testDB.getAtom(trusts, grounding)
			if (atom instanceof RandomVariableAtom) {
				atom.setValue(0.0)
			}
		}
		/* For discrete MRFs, "MPE" inference will actually perform marginal inference */
		MPEInference mpe = new MPEInference(m, testDB, config)
		FullInferenceResult result = mpe.mpeInference()
		testDB.close()

		/*
		 * Evaluation
		 */
		Database resultsDB = data.getDatabase(dummy2, write_te)
		def comparator = new SimpleRankingComparator(resultsDB)
		def groundTruthDB = data.getDatabase(labels_te, [trusts] as Set)
		comparator.setBaseline(groundTruthDB)

		def metrics = [RankingScore.AUPRC, RankingScore.NegAUPRC, RankingScore.AreaROC]
		double [] score = new double[metrics.size()]

		for (int i = 0; i < metrics.size(); i++) {
			comparator.setRankingScore(metrics.get(i))
			score[i] = comparator.compare(trusts)
		}
		System.out.println("Area under positive-class PR curve: " + score[0])
		System.out.println("Area under negative-class PR curve: " + score[1])
		System.out.println("Area under ROC curve: " + score[2])

		results.get(configIndex).add(fold, score)

		/*
		 * Output the inferred edge labels.
		 */
		File outFile;
		if (evalType.equals("varyingObsRatio") || evalType.equals("varyingDiscardedFeats")) {
			outFile = new File(BASEDIR + "/results/" + evalType + "/" + (noObservationsForTesting
					? "noObservationsForTesting" : "") + "/" + config_test
				+ "/predictions_" + (joint ? "joint_discard-top-" + numTopFeatsToDiscard
				+ "-feats_binned" : "structure-only") + "_obsRatio=" + obsRatio + ".txt");
		} else if (evalType.equals("lexiconBased")) {
			outFile = new File(BASEDIR+"/results/lexiconBased/"+config_test
				+ "/predictions_" + (joint ? "joint_binned" : "structure-only") + "_posCutoff="
				+ posCutoff + "_negCutoff=" + negCutoff + ".txt");
		} else {
			throw new IllegalArgumentException(evalType + " is not a legal evalType value");
		}
		PrintStream out = new PrintStream(outFile);
		for (GroundAtom atom : Queries.getAllAtoms(resultsDB, trusts)) {
			GroundTerm[] args = atom.getArguments()
			out.format("%s\t%s\t%s\n", args[0].toString(), args[1].toString(), atom.getValue())
		}
		out.close()

		resultsDB.close()
		groundTruthDB.close()
	}
	trainDB.close()
	labelsDB.close()
}

for (int configIndex = 0; configIndex < configs.size(); configIndex++) {
	def methodStats = results.get(configIndex)
	configName = configs.get(configIndex).getString("name", "");
	sum = new double[3];
	sumSq = new double[3];
	for (int fold = 0; fold < folds; fold++) {
		def score = methodStats.get(fold)
		for (int i = 0; i < 3; i++) {
			sum[i] += score[i];
			sumSq[i] += score[i] * score[i];
		}
		System.out.println("Method " + configName + ", fold " + fold +", auprc positive: "
				+ score[0] + ", negative: " + score[1] + ", auROC: " + score[2])
	}

	mean = new double[3];
	variance = new double[3];
	for (int i = 0; i < 3; i++) {
		mean[i] = sum[i] / folds;
		variance[i] = sumSq[i] / folds - mean[i] * mean[i];
	}

	System.out.println();
	System.out.println("Method " + configName + ", auprc positive: (mean/variance) "
			+ mean[0] + "  /  " + variance[0] );
	System.out.println("Method " + configName + ", auprc negative: (mean/variance) "
			+ mean[1] + "  /  " + variance[1] );
	System.out.println("Method " + configName + ", auROC: (mean/variance) "
			+ mean[2] + "  /  " + variance[2] );
	System.out.println();
}


public void learn(Model m, Database db, Database labelsDB, ConfigBundle config, Logger log) {
	switch(config.getString("learningmethod", "")) {
		case "MLE":
			MaxLikelihoodMPE mle = new MaxLikelihoodMPE(m, db, labelsDB, config)
			mle.learn()
			break
		case "MPLE":
			MaxPseudoLikelihood mple = new MaxPseudoLikelihood(m, db, labelsDB, config)
			mple.learn()
			break
		case "MM":
			MaxMargin mm = new MaxMargin(m, db, labelsDB, config)
			mm.learn()
			break
		default:
			throw new IllegalArgumentException("Unrecognized method.");
	}
}
