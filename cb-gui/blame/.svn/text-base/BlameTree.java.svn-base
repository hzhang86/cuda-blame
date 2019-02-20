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
import java.awt.event.MouseAdapter;
import java.awt.event.MouseListener;
import java.awt.event.MouseEvent;

import java.util.Vector;

//import java.net.URL;
import java.util.Iterator;
//import java.io.IOException;
import java.awt.Dimension;
import java.awt.GridLayout;

public class BlameTree extends JPanel implements MouseListener {
    public  JEditorPane metadataPane;
    public JTree tree;
    public NodeSideInfoTree nsit;
    
    //private URL helpURL;
    //private static boolean DEBUG = false;
    
    private BlameContainer bc;
    

    //Optionally play with line styles.  Possible values are
    //"Angled" (the default), "Horizontal", and "None".
    private static boolean playWithLineStyle = false;
    private static String lineStyle = "Horizontal";
    
    //Optionally set the look and feel.
    private static boolean useSystemLookAndFeel = false;

    public BlameTree(BlameContainer bc) {
    	
        super(new GridLayout(1,0));

        this.bc = bc;
        nsit = new NodeSideInfoTree(bc);
        
        //Create the nodes.
        DefaultMutableTreeNode top =
            new DefaultMutableTreeNode("Program Blame");
        createNodes(top);

        //Create a tree that allows one selection at a time.
        tree = new JTree(top);
        tree.getSelectionModel().setSelectionMode
                (TreeSelectionModel.SINGLE_TREE_SELECTION);

        /*
        tree.setCellRenderer(new DefaultTreeCellRenderer()
        {
             public Component getTreeCellRendererComponent(JTree pTree,
                 Object pValue, boolean pIsSelected, boolean pIsExpanded,
                 boolean pIsLeaf, int pRow, boolean pHasFocus)
             {
        DefaultMutableTreeNode node = (DefaultMutableTreeNode)pValue;
        super.getTreeCellRendererComponent(pTree, pValue, pIsSelected,
                     pIsExpanded, pIsLeaf, pRow, pHasFocus);
                 if (node.isRoot())
           setBackgroundSelectionColor(Color.red);
        else if (node.getChildCount() > 0)
           setBackgroundSelectionColor(Color.yellow);
        else if (pIsLeaf)
           setBackgroundSelectionColor(Color.green);
        return (this);
    }
        });*/

        
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
    
    private void createNodes(DefaultMutableTreeNode top) {
        
    	DefaultMutableTreeNode blamePointExplicit = null;
    	DefaultMutableTreeNode blamePointUserDefined = null;
    	DefaultMutableTreeNode nonBlamePoint = null;
    	
    	blamePointExplicit = new DefaultMutableTreeNode("Explicit Blame Points");
        top.add(blamePointExplicit);
        blamePointUserDefined = new DefaultMutableTreeNode("User Defined Blame Points");
        top.add(blamePointUserDefined);
        nonBlamePoint = new DefaultMutableTreeNode("Non Blame Points");
        top.add(nonBlamePoint);
    	
        DefaultMutableTreeNode functions = null;
    	DefaultMutableTreeNode variables = null;
    
    	
    	DefaultMutableTreeNode evTree = null;
    	DefaultMutableTreeNode epTree = null;
    	DefaultMutableTreeNode eoTree = null;
    		
    	
		//Iterator it = bc.getBlamedFunctions().values().iterator();
		
    	Iterator it = bc.getAllFunctions().values().iterator();
    	while (it.hasNext())
		{
			BlameFunction bf = (BlameFunction) it.next();
			functions = new DefaultMutableTreeNode(bf);
			if (bf.getBlamePointType() == BlameFunction.EXPLICIT_BLAME_POINT)
			{
				blamePointExplicit.add(functions);
				
			 	//evTree = new DefaultMutableTreeNode("Exit Variables");
		        //functions.add(evTree);
		        epTree = new DefaultMutableTreeNode("Local Variables");
		        functions.add(epTree);
		        eoTree = new DefaultMutableTreeNode("Output");
		        functions.add(eoTree);
				
				Vector<ExitProgram> epVec = new Vector<ExitProgram>(bf.getExitPrograms().values());
				Collections.sort(epVec);
				
				Iterator<ExitProgram> ep_it = epVec.iterator();
				while (ep_it.hasNext())
				{
					ExitProgram ep = (ExitProgram)ep_it.next();					
					variables = new DefaultMutableTreeNode(ep);
					epTree.add(variables);
											
					recursiveAddFields(variables, ep);
			
				}
				
				
				//Vector<ExitOutput> eoVec = (Vector<ExitOutput>) bf.getExitOutputs().values();
				//Iterator<ExitOutput> eo_it = eoVec.iterator();
				Iterator<ExitOutput> eo_it = bf.getExitOutputs().values().iterator();

				
				while (eo_it.hasNext())
				{
					ExitOutput eo = (ExitOutput)eo_it.next();
					variables = new DefaultMutableTreeNode(eo);
					eoTree.add(variables);
				}
				
				
			} 
			else if (bf.getBlamePointType() == BlameFunction.IMPLICIT_BLAME_POINT)
			{
				blamePointUserDefined.add(functions);
				
			 	evTree = new DefaultMutableTreeNode("Exit Variables");
		        functions.add(evTree);
		        epTree = new DefaultMutableTreeNode("Local Variables");
		        functions.add(epTree);
		        eoTree = new DefaultMutableTreeNode("Output");
		        functions.add(eoTree);
				
				
				Vector<ExitVariable> evVec = new Vector<ExitVariable>(bf.getExitVariables().values());
				Collections.sort(evVec);
				Iterator<ExitVariable> ev_it = evVec.iterator();
				while (ev_it.hasNext())
				{
					ExitVariable ev = (ExitVariable)ev_it.next();
					if (ev.getHierName().indexOf('-') < 0)
					{
						variables = new DefaultMutableTreeNode(ev);
					
						evTree.add(variables);
						recursiveAddFields(variables, ev);
					}
				}
				
				
				Vector<ExitProgram> epVec = new Vector<ExitProgram>(bf.getExitPrograms().values());
				
				Collections.sort(epVec);
				
				Iterator<ExitProgram> ep_it = epVec.iterator();
				while (ep_it.hasNext())
				{
					ExitProgram ep = (ExitProgram)ep_it.next();
					variables = new DefaultMutableTreeNode(ep);
					epTree.add(variables);
					
					
					recursiveAddFields(variables, ep);
				}
				
				
				//Vector<ExitOutput> eoVec = (Vector<ExitOutput>) bf.getExitOutputs().values();
				//Iterator<ExitOutput> eo_it = eoVec.iterator();
				Iterator<ExitOutput> eo_it = bf.getExitOutputs().values().iterator();

				
				while (eo_it.hasNext())
				{
					ExitOutput eo = (ExitOutput)eo_it.next();
					variables = new DefaultMutableTreeNode(eo);
					eoTree.add(variables);
				}
				
				
			}
			// Not a blame Point
			else
			{
				nonBlamePoint.add(functions);
				
			 	evTree = new DefaultMutableTreeNode("Exit Variables");
		        functions.add(evTree);
		        //epTree = new DefaultMutableTreeNode("Local Variables");
		        //functions.add(epTree);
		        eoTree = new DefaultMutableTreeNode("Output");
		        functions.add(eoTree);
				
				
				Vector<ExitVariable> evVec = new Vector<ExitVariable>(bf.getExitVariables().values());				
				Collections.sort(evVec);
				
				Iterator<ExitVariable> ev_it = evVec.iterator();
				while (ev_it.hasNext())
				{
					ExitVariable ev = (ExitVariable)ev_it.next();
					variables = new DefaultMutableTreeNode(ev);
					
					evTree.add(variables);
					
					recursiveAddFields(variables, ev);

				}
				
				//Vector<ExitOutput> eoVec = (Vector<ExitOutput>) bf.getExitOutputs().values();
				//Iterator<ExitOutput> eo_it = eoVec.iterator();
				
				Iterator<ExitOutput> eo_it = bf.getExitOutputs().values().iterator();
				
				while (eo_it.hasNext())
				{
					ExitOutput eo = (ExitOutput)eo_it.next();
					variables = new DefaultMutableTreeNode(eo);
					eoTree.add(variables);
				}
				
				
			}
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
