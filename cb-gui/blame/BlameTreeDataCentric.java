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

public class BlameTreeDataCentric extends JPanel implements MouseListener {
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
    

    //Optionally play with line styles.  Possible values are
    //"Angled" (the default), "Horizontal", and "None".
    private static boolean playWithLineStyle = false;
    private static String lineStyle = "Horizontal";
    
    //Optionally set the look and feel.
    private static boolean useSystemLookAndFeel = false;

    public BlameTreeDataCentric(BlameContainer bc) {
    	
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

			if (node.getLevel() >= 1)
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
    	//added by Hui 01/26/16 for test //
        System.out.println(es.getName()+" has "+esVec.size()+" fields");
		//Iterator<ExitSuper> es_it = es.getFields().values().iterator();
		while (es_it.hasNext())
		{
			ExitSuper field = (ExitSuper)es_it.next();
            ////added by Hui 01/26/16 for test////
            System.out.println("Adding field node " + field.getName() + " to parent node " + es.getName());
            //////////////////////////////////////
			DefaultMutableTreeNode fields = new DefaultMutableTreeNode(field);
			parent.add(fields);
			if (field.getFields().size() > 0)
				recursiveAddFields(fields, field);
			
		}
    }
    
    private void createNodes(DefaultMutableTreeNode top) {
      
    	DefaultMutableTreeNode variables = null;
        			
    	Vector<ExitSuper> epVec = bc.getAllLocalVariables();
    	Vector<ExitSuper> globalVec = bc.getAllGlobalVariables();
    	
    	System.out.println("Size of epVec is " + epVec.size());
    	System.out.println("Size of globalVec is " + globalVec.size());
    	
    	epVec.addAll(globalVec);
    	System.out.println("Size of merged vector is " + epVec.size());
    	
    	Collections.sort(epVec);
				
		Iterator<ExitSuper> ep_it = epVec.iterator();
		while (ep_it.hasNext())
		{
			ExitSuper ep = (ExitSuper)ep_it.next();					
			variables = new DefaultMutableTreeNode(ep);
			top.add(variables);
											
			recursiveAddFields(variables, ep);
			
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
        frame.add(this);
        
        //Display the window.
        frame.pack();
        frame.setVisible(true);
    }
}
