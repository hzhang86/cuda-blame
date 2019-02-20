/*
 *  Copyright 2014-2017 Hui Zhang
 *  Previous contribution by Nick Rutar 
 *  All rights reserved.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

package blame;

//import java.util.Comparator;
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
import javax.swing.tree.DefaultTreeModel;
import javax.swing.tree.TreeSelectionModel;
//import javax.swing.event.TreeSelectionEvent;
//import javax.swing.event.TreeSelectionListener;

import java.awt.event.*;

//import java.awt.event.WindowListener;
//import java.awt.event.WindowAdapter;

import java.util.Vector;

//import java.net.URL;
import java.util.Iterator;
//import java.io.IOException;
import java.awt.Dimension;
import java.awt.GridLayout;

public class NodeSideInfoTree extends JPanel {

	   public  JEditorPane metadataPane;
	    public JTree tree;
	    //private URL helpURL;
	    //private static boolean DEBUG = false;
	    
	    //private BlameContainer bc;
	    DefaultMutableTreeNode top;
		DefaultMutableTreeNode nodes;
    	DefaultMutableTreeNode instances;
    	
    	private boolean active;
	    
	    public boolean isActive() {
			return active;
		}

		public void setActive(boolean active) {
			this.active = active;
		}

		ExitSuper es;

	    public ExitSuper getEs() {
			return es;
		}

		public void setEs(ExitSuper es) {
			this.es = es;
		}

		//Optionally play with line styles.  Possible values are
	    //"Angled" (the default), "Horizontal", and "None".
	    private static boolean playWithLineStyle = false;
	    private static String lineStyle = "Horizontal";
	    
	    //Optionally set the look and feel.
	    private static boolean useSystemLookAndFeel = false;

	    public NodeSideInfoTree()
	    {
	    	tree = new JTree();
	    	
	    }
	    
	    public NodeSideInfoTree(BlameContainer bc) {
	    	
	        super(new GridLayout(1,0));

	        //this.bc = bc;
	        
	        // DefaultMutableTreeNode 
	        top =    new DefaultMutableTreeNode("Nodes");
	        nodes = null;
	        instances = null;
	        
	        //Nodes will be created once clicked on a variable (es)
            //createdNodes called in BlameTreeDataCentric

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

	       
	        //Add the split pane to this panel.
	        add(splitPane);
	    }


	    public void createNodes(ExitSuper ep) 
	    { 	
	    	
	    	active = true;
	    	tree.removeAll();
	        
	    	
	        DefaultTreeModel model = (DefaultTreeModel) tree.getModel();
	        
	        
	        top =    new DefaultMutableTreeNode("Nodes");
	        model.setRoot(top);
	        
	       // tree.get
	        
	        //top.removeAllChildren();
	    
	    	nodes = null;
	        instances = null;
	    	
	    	es = ep;
			
			Vector<NodeInstance> niVec = new Vector<NodeInstance>(ep.getNodeInstances().values());
			Iterator ni_it = niVec.iterator();
			while (ni_it.hasNext())
			{
				NodeInstance ni = (NodeInstance)ni_it.next();
				nodes = new DefaultMutableTreeNode(ni);
				//System.out.println("Adding node for " + ni.getNodeName());
				top.add(nodes);
				
				//Vector<VariableInstance> viVec = ni.getVarInstances();
				Iterator vi_it = ni.getVarInstances().values().iterator();
				while (vi_it.hasNext())
				{
					VariableInstance vi = (VariableInstance)vi_it.next();
					instances = new DefaultMutableTreeNode(vi);
					//System.out.println("Adding node for Instance " + vi.getInst().getInstanceNum());
					nodes.add(instances);
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
	        JFrame frame = new JFrame("Node Information for Variable");
	        //frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	        
	        // Exit app when frame is closed.
	        frame.addWindowListener(new WindowAdapter() {
	            public void windowClosing(WindowEvent ev) {
	                //System.exit(0);
	            	active = false;
	            }
	        });

	        //Add content to the window.
	        //frame.add(new BlameTree(bc));
	        frame.add(this);
	        
	        //Display the window.
	        frame.pack();
	        frame.setVisible(true);
	    }
}
	
	
	

