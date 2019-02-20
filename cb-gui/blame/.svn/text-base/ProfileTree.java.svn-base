package blame;

//import java.util.Comparator;
import java.util.Collections;

import javax.swing.JEditorPane;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.UIManager;

//import java.awt.*;
//import javax.swing.*;
//import javax.swing.tree.*;

import javax.swing.JTree;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreeSelectionModel;
//import javax.swing.event.TreeSelectionEvent;
//import javax.swing.event.TreeSelectionListener;
import java.awt.event.MouseListener;
import java.awt.event.MouseEvent;

import java.util.Vector;

//import java.net.URL;
import java.util.Iterator;
//import java.io.IOException;
import java.awt.Dimension;
import java.awt.GridLayout;

public class ProfileTree extends JPanel implements MouseListener {
    /**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public  JEditorPane metadataPane;
    public JTree tree;
    public NodeSideInfoTree nsit;
    
    //private URL helpURL;
    //private static boolean DEBUG = false;
    
    private BlameContainer bc;
    private profiler.ProfilerData pd;
    

    //Optionally play with line styles.  Possible values are
    //"Angled" (the default), "Horizontal", and "None".
    private static boolean playWithLineStyle = false;
    private static String lineStyle = "Horizontal";
    
    //Optionally set the look and feel.
    private static boolean useSystemLookAndFeel = false;

    public ProfileTree(profiler.ProfilerData pd) {
    	
        super(new GridLayout(1,0));

        this.pd = pd;
        //this.bc = bc;
        //nsit = new NodeSideInfoTree(bc);
        
        //Create the nodes.
        DefaultMutableTreeNode top =
            new DefaultMutableTreeNode("Program Functions");
        createNodes(top);

        //Create a tree that allows one selection at a time.
        tree = new JTree(top);
        tree.getSelectionModel().setSelectionMode
                (TreeSelectionModel.SINGLE_TREE_SELECTION);
     
        //Listen for when the selection changes.
        //tree.addTreeSelectionListener(this);

        if (playWithLineStyle) {
            //System.out.println("line style = " + lineStyle);
            tree.putClientProperty("JTree.lineStyle", lineStyle);
        }

        //Create the scroll pane and add the tree to it. 
        JScrollPane treeView = new JScrollPane(tree);

        //Create the HTML viewing pane.
        metadataPane = new JEditorPane();
        metadataPane.setEditable(false);
        //initHelp();
        JScrollPane htmlView = new JScrollPane(metadataPane);

        //Add the scroll panes to a split pane.
        JSplitPane splitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT);
        splitPane.setTopComponent(treeView);
        splitPane.setBottomComponent(htmlView);

        Dimension minimumSize = new Dimension(100, 50);
        htmlView.setMinimumSize(minimumSize);
        treeView.setMinimumSize(minimumSize);
        splitPane.setDividerLocation(500); 
        splitPane.setPreferredSize(new Dimension(400, 600));

        tree.addMouseListener(this);
        //Add the split pane to this panel.
        add(splitPane);
    }

    public void mousePressed(MouseEvent e) {
    }
    
    public void mouseReleased(MouseEvent e) {
    }
    
    public void mouseEntered(MouseEvent e) {
    }
    
    public void mouseExited(MouseEvent e) {
    }
    
    public void mouseClicked(MouseEvent e) {
    	
        System.out.println("Mouse clicked (# of clicks: "
                + e.getClickCount() + ")");
                
        if (e.getClickCount() == 2)
        {
			DefaultMutableTreeNode node = (DefaultMutableTreeNode)
			tree.getLastSelectedPathComponent();

			if (node == null) return;

			Object nodeInfo = node.getUserObject();

			if (node.getLevel() >= 3)
			{
				System.out.println("Clicking on a Node");
				ExitSuper es = (ExitSuper) nodeInfo;
				System.out.println("Node for " + es.getName());
				//nsit = new NodeSideInfoTree(bc);
				nsit.createNodes(es);
				
				nsit.setActive(true);
				nsit.createAndShowGUI();
				//System.out.println("Here");
				//nsit.setActive(false);
			}
        }
    }

    
    
    private void recursiveAddFields(DefaultMutableTreeNode parent, ExitSuper es)
    {
		Vector<ExitSuper> esVec = new Vector<ExitSuper>(es.getFields().values());
		Collections.sort(esVec);
		
		Iterator<ExitSuper> es_it = esVec.iterator();
    	
		//Iterator<ExitSuper> es_it = es.getFields().values().iterator();
		while (es_it.hasNext())
		{
			ExitSuper field = (ExitSuper)es_it.next();
			DefaultMutableTreeNode fields = new DefaultMutableTreeNode(field);
			parent.add(fields);
			if (field.getFields().size() > 0)
				recursiveAddFields(fields, field);
			
		}
    }
    
    
    private void recursiveAddFunctions(DefaultMutableTreeNode parent, profiler.ProfilerFunction pf)
    {
    	Vector<profiler.ProfilerFunction> pfVec = pf.getFuncDescendants();
    	//Collections.sort(pfVec);
    	
    	Iterator<profiler.ProfilerFunction> pf_it = pfVec.iterator();
    	while (pf_it.hasNext())
    	{
    		profiler.ProfilerFunction pf2 = (profiler.ProfilerFunction) pf_it.next();
    		DefaultMutableTreeNode fields = new DefaultMutableTreeNode(pf2);
    		parent.add(fields);
    		if (pf2.getFuncDescendants().size() > 0)
    			recursiveAddFunctions(fields, pf2);
 
    	}
    	
    }
    
    
    private void createNodes(DefaultMutableTreeNode top) 
    {
      
    	DefaultMutableTreeNode functions = null;
    	//functions = new DefaultMutableTreeNode("main");
    	
    	//profiler.ProfilerFunction mainFunc = pd.getAllFunctions().get("MAIN__");
    	
    	//System.err.println("Size of main is " + mainFunc.getFuncDescendants().size());
    	
    	//top.add(functions);
    	
    	//recursiveAddFunctions(functions, mainFunc);
    	
    	// Should probably be smarter about this, hope we never run into a call path over 100
    	DefaultMutableTreeNode[] tNodes = new DefaultMutableTreeNode[100];
    	
		Iterator<profiler.ProfilerFunction> it = pd.getSortedFunctions(true).values().iterator();
		
		//int prevLevel = 0;
		//int currentLevel = 0;
		
		while (it.hasNext())
		{
			profiler.ProfilerFunction bf = (profiler.ProfilerFunction) it.next();	
			
			int numPeriods = bf.getNumPeriods();
			//currentLevel = numPeriods;
			
			functions = new DefaultMutableTreeNode(bf);
			
			if (numPeriods == 0)
			{
				top.add(functions);
				tNodes[0] = functions;
			}
			else
			{
				tNodes[numPeriods-1].add(functions);
				tNodes[numPeriods] = functions;
			}
			
			//prevLevel = currentLevel;
			
			//pStream.println(bf);
				
			
		}
  
    	
    	
    }
        
    
    
    
    
    /**
     * Create the GUI and show it.  For thread safety,
     * this method should be invoked from the
     * event dispatch thread.
     */
    public void createAndShowGUI() {
        if (useSystemLookAndFeel) {
            try {
                UIManager.setLookAndFeel(
                    UIManager.getSystemLookAndFeelClassName());
            } catch (Exception e) {
                System.err.println("Couldn't use system look and feel.");
            }
        }

        //Create and set up the window.
        JFrame frame = new JFrame("Blame Points and Variables");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        

        //Add content to the window.
        //frame.add(new BlameTree(bc));
        frame.add(this);
        
        //Display the window.
        frame.pack();
        frame.setVisible(true);
    }
}
