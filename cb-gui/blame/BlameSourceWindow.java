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



import javax.swing.*;
import java.awt.Component;
import java.awt.Container;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.util.Vector;
import java.util.Iterator;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JScrollPane;
import javax.swing.ListCellRenderer;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.event.TreeSelectionEvent;
import javax.swing.event.TreeSelectionListener;
import javax.swing.tree.DefaultMutableTreeNode;

import java.awt.Color;



public class BlameSourceWindow extends JFrame {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private JList stackNameList;
	private JList sourceBodyList;
	//private JList variableNameList, programNameList;
	//private JScrollPane jBody;
	//private JScrollPane pBody;
	private JScrollPane sourceBody;
	private JScrollPane jMessages;
	private GridBagLayout gbl;
	private GridBagConstraints gbc;

	//private BlameNodeWindow localBNW;
	//private VariableMetaData localVMD;
	private BlameTreeBlamePoints localBT;
	private NodeSideInfoTree localNSIT;
	private NodeSideInfoTree localNSIT2;
	
	short whichNSIT;



	//private SourceContainer sourceContainer;

	//private Vector<String> stackNames;
	//private Vector<String> variableNames, programNames;
	//private Vector<String> sourceBody;

	//private JLabel variableLabel, programLabel;
	private JLabel stackFrameLabel;
	private JLabel bodyLabel;
	private Container c;



	public BlameSourceWindow(BlameContainer bc, BlameNodeWindow bnw, 
			VariableMetaData vmd, BlameTreeBlamePoints bt, BlameTreeDataCentric btdc)
	{
		super("Source Files");

		//localBNW = bnw;
		//localVMD = vmd;
		localBT =  bt;
		localNSIT = bt.nsit;
		localNSIT2 = btdc.nsit;
		
		whichNSIT = 0;

		ParentData node0 = bc.getNodes().elementAt(0);
		Instance instance0 = new Instance(0, "NUUL");

		if (node0 != null)
		{
			Vector<Instance> instances0 = node0.instances;
			if (instances0!= null)
			instance0 = instances0.elementAt(0);
		}
		else
		{
			System.err.println("Node 0 data is NULL");
			System.exit(0);
		}

		//Vector<StackFrame> sFrames0 = instance0.getStackFrames();

		//instance0.print();

		// Names of source files, will change as instance changes
		//variableNames = new Vector<String>();
		//programNames = new Vector<String>();

		// The function names will be changed on the fly as the source name changes
		//stackNames = new Vector<String>();

		// Actual lines of code
		//sourceBody = new Vector<String>();

		// Adds all of the source names to the source Vector
		/*
		for (int a = 0; a < sourceContainer.getSourceFiles().size(); a++)
		{
			sourceNames.add( ((SourceFile) sourceContainer.getSourceFiles().get(a)).getFileName() );
		}
		 */

		// Sets up the GridBagLayout Layout 
		///////////////////////////////////
		c = getContentPane();
		gbl = new GridBagLayout();
		c.setLayout(gbl);

		gbc = new GridBagConstraints();



		//Message List JList and Scroll Pane for list of Source Files
		/////////////////////////////////////


		//stackNameList = new JList( sFrames0 );
		stackNameList = new JList(new Vector<String>());
		stackNameList.setFixedCellWidth(500);
		stackNameList.setVisibleRowCount(5);
		stackNameList.setSelectedIndex(0);
		jMessages = new JScrollPane(stackNameList);

		gbc.weightx = 1000;
		gbc.weighty = 1;
		gbc.fill = GridBagConstraints.BOTH;
		addComponent(jMessages, 1, 0, 1, 1);

		//Message Body JList and Scroll Pane for Source Code Body
		////////////////////////////////////

		/*
		if (sourceContainer.getSourceFiles().size() > 0)
			sourceBodyList = new JList( ( (SourceFile) sourceContainer.getSourceFiles().get(0)).getLines() );
		else*/

		sourceBodyList = new JList();
		sourceBodyList.setSelectedIndex(0);
		sourceBodyList.setVisibleRowCount(20);
		sourceBodyList.setFixedCellHeight(12);
		sourceBodyList.setFixedCellWidth(500);
		ListCellRenderer renderer = new ComplexCellRenderer();
		sourceBodyList.setCellRenderer(renderer);

		sourceBody = new JScrollPane(sourceBodyList);
		gbc.weighty = 100;
		gbc.weightx = 100;
		gbc.fill = GridBagConstraints.BOTH;

		addComponent(sourceBody, 3, 0, 3, 1);
		//gbc.fill = GridBagConstraints.NONE;


		/*
		//Message Body JList and Scroll Pane for Exit Variables
		////////////////////////////////////
		variableNameList = new JList( variableNames );
		gbc.weighty = 10;
		variableNameList.setVisibleRowCount(20);
		variableNameList.setFixedCellHeight(12);
		variableNameList.setFixedCellWidth(500);
		variableNameList.setSelectedIndex(0);
		jBody = new JScrollPane(variableNameList);
		//gbc.fill = GridBagConstraints.BOTH;
		addComponent(jBody, 1, 1, 1, 1);

		// Message body JList and Scroll Pane for Exit Programs
		programNameList = new JList( programNames );
		gbc.weighty = 20;
		programNameList.setVisibleRowCount(20);
		programNameList.setFixedCellHeight(12);
		programNameList.setFixedCellWidth(500);
		programNameList.setSelectedIndex(0);
		pBody = new JScrollPane(programNameList);
		//gbc.fill = GridBagConstraints.BOTH;
		addComponent(pBody, 1, 2, 1, 1);
		 */

		gbc.weightx = 0;
		gbc.weighty = 0;
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.NORTHWEST;

		//Subject & Time Label
		//////////////////////
		stackFrameLabel = new JLabel("Stack Frame/Source Module:");
		addComponent(stackFrameLabel, 0, 0, 1, 1);

		/*
		variableLabel = new JLabel("Exit Variable");
		addComponent(variableLabel, 0 , 1, 1, 1);

		programLabel = new JLabel("Local Variable");
		addComponent(programLabel, 0, 2, 1, 1);
		 */

		//Message Body Label
		////////////////////////////////////
		bodyLabel = new JLabel("Source");
		addComponent(bodyLabel, 2, 0, 3, 1);



		sourceBodyList.addListSelectionListener(
				new ListSelectionListener() {
					public void valueChanged( ListSelectionEvent e)
					{
						if ( e.getValueIsAdjusting() == false)
						{
							// Get the current stack frame
							int index = stackNameList.getSelectedIndex();
							
							
							//stackNameList.setSelectedIndex(index);
							//System.out.println(index);
							
							if (index >= 0)
								return;
							
							DefaultMutableTreeNode node = (DefaultMutableTreeNode)
							localBT.tree.getLastSelectedPathComponent();

							if (node == null) return;

							Object nodeInfo = node.getUserObject();

							
							if (node.getLevel() < 5)
							{
								// Worry about this later
								return;
							}


							if (node.getLevel() == 2)
							{
								VariableInstance vi = (VariableInstance)nodeInfo;
								Instance inst = vi.getInst();
								int instIndex = inst.getInstanceNum();

								if (instIndex < 0)
									inst = new Instance(0, "NULL");

								Vector<StackFrame> sFrames0 = inst.getStackFrames();

								if (sFrames0.size() > 0)
								{

									while (node.getLevel() != 2)
										node = (DefaultMutableTreeNode) node.getParent();

									BlameFunction bf = (BlameFunction)node.getUserObject();

									Iterator<StackFrame> it = sFrames0.iterator();
									StackFrame sf = sFrames0.get(0);
									int sfIndex = 0;
									while (it.hasNext())
									{
										sf = (StackFrame)it.next();
										if (sf.getBf().equals(bf))
											break;
										sfIndex++;
									}

									stackNameList.setSelectedIndex(sfIndex);
									//stackNameList.setListData(sFrames0);


								}
							}
							
						}
					}
				}
		);

		stackNameList.addListSelectionListener(
				new ListSelectionListener() {
					public void valueChanged( ListSelectionEvent e)
					{
						if ( e.getValueIsAdjusting() == false)
						{

							
							DefaultMutableTreeNode node = (DefaultMutableTreeNode)
							localBT.tree.getLastSelectedPathComponent();
							
							//if (node.getLevel() < 5)
								//return;
							// Get the current stack frame

							int index = stackNameList.getSelectedIndex();

							//System.out.println("Selected Value is " + index);

							
							
							if (index >= 0)
							{
								if (!(stackNameList.getSelectedValue() instanceof StackFrame))
									return;
								

								StackFrame sf = (StackFrame) stackNameList.getSelectedValue();

									

								// From stack frame get variable list
								BlameFunction bf = sf.bf;


								//Collection<ExitVariable> v = bf.getExitVariables().values();
								//v.toArray();

								//variableNameList.setListData(bf.getExitVariables().values().toArray());
								//programNameList.setListData(bf.getExitPrograms().values().toArray());

								// From stack frame get Source file

								SourceFile sFile = (SourceFile) SourceContainer.sourceFiles.get(sf.moduleName);
								sourceBodyList.setListData( sFile.getLines() );

								//sourceBodyList.setCellRenderer(new MyCellRenderer());
								sourceBodyList.setSelectedIndex(sf.lineNumber-1);
								sourceBodyList.ensureIndexIsVisible(sf.lineNumber-1);

								sourceBody.revalidate();
								sourceBody.repaint();

								//jBody.revalidate();
								//jBody.repaint();

								//pBody.revalidate();
								//pBody.repaint();
							}
						}
					}
				}
		);

		bt.nsit.tree.addTreeSelectionListener(
				new TreeSelectionListener() {
					public void valueChanged(TreeSelectionEvent e)
					{
						DefaultMutableTreeNode node = (DefaultMutableTreeNode)
						localNSIT.tree.getLastSelectedPathComponent();

						whichNSIT = 0;
						
						if (node == null) return;

						System.out.println("in nsit listener");
						
						Object nodeInfo = node.getUserObject();

				

						if (node.getLevel() == 1 )
						{
							stackFrameLabel.setText("");
							localNSIT.metadataPane.setText("");
							sourceBodyList.setListData(new Vector<String>());
							stackNameList.setListData(new Vector<String>());
						}
					
						if (node.getLevel() == 2)
						{
							stackFrameLabel.setText("Stack Frames");
							VariableInstance vi = (VariableInstance)nodeInfo;
							Instance inst = vi.getInst();
							int index = inst.getInstanceNum();

							if (index < 0)
								inst = new Instance(0, "NULL");

							Vector<StackFrame> sFrames0 = inst.getStackFrames();

							if (sFrames0.size() > 0)
							{
								StackFrame sf = sFrames0.get(0);
								int sfIndex = 0;
								
								stackNameList.setSelectedIndex(sfIndex);
								stackNameList.setListData(sFrames0);


								SourceFile sFile = (SourceFile) SourceContainer.sourceFiles.get(sf.moduleName);
								sourceBodyList.setListData( sFile.getLines() );

								//sourceBodyList.setCellRenderer(new MyCellRenderer());
								sourceBodyList.setSelectedIndex(sf.lineNumber-1);
								sourceBodyList.ensureIndexIsVisible(sf.lineNumber-1);

							}
							else
							{
								sourceBodyList.setListData(new Vector<String>());
								stackNameList.setListData(new Vector<String>());
							}

							jMessages.revalidate();
							jMessages.repaint();

							sourceBody.revalidate();
							sourceBody.repaint();
						}
					}
				}
		);


		btdc.nsit.tree.addTreeSelectionListener(
				new TreeSelectionListener() {
					public void valueChanged(TreeSelectionEvent e)
					{
						DefaultMutableTreeNode node = (DefaultMutableTreeNode)
						localNSIT2.tree.getLastSelectedPathComponent();

						whichNSIT = 1;
						
						if (node == null) return;

						System.out.println("in nsit listener");
						
						Object nodeInfo = node.getUserObject();

				

						if (node.getLevel() == 1 )
						{
							stackFrameLabel.setText("");
							localNSIT2.metadataPane.setText("");
							sourceBodyList.setListData(new Vector<String>());
							stackNameList.setListData(new Vector<String>());
						}
					
						if (node.getLevel() == 2)
						{
							stackFrameLabel.setText("Stack Frames");
							VariableInstance vi = (VariableInstance)nodeInfo;
							Instance inst = vi.getInst();
							int index = inst.getInstanceNum();

							if (index < 0)
								inst = new Instance(0, "NULL");

							Vector<StackFrame> sFrames0 = inst.getStackFrames();

							if (sFrames0.size() > 0)
							{
								StackFrame sf = sFrames0.get(0);
								int sfIndex = 0;
								
								stackNameList.setSelectedIndex(sfIndex);
								stackNameList.setListData(sFrames0);


								SourceFile sFile = (SourceFile) SourceContainer.sourceFiles.get(sf.moduleName);
								sourceBodyList.setListData( sFile.getLines() );

								//sourceBodyList.setCellRenderer(new MyCellRenderer());
								sourceBodyList.setSelectedIndex(sf.lineNumber-1);
								sourceBodyList.ensureIndexIsVisible(sf.lineNumber-1);

							}
							else
							{
								sourceBodyList.setListData(new Vector<String>());
								stackNameList.setListData(new Vector<String>());
							}

							jMessages.revalidate();
							jMessages.repaint();

							sourceBody.revalidate();
							sourceBody.repaint();
						}
					}
				}
		);
		

		bt.tree.addTreeSelectionListener(
				new TreeSelectionListener() {
					public void valueChanged(TreeSelectionEvent e)
					{
						//System.out.println("Change");

						DefaultMutableTreeNode node = (DefaultMutableTreeNode)
						localBT.tree.getLastSelectedPathComponent();

						if (node == null) return;

						Object nodeInfo = node.getUserObject();

						if (node.getLevel() < 4)
						{
							localBT.metadataPane.setText("");
						}
						if (node.getLevel() >= 4)
						{
							stackFrameLabel.setText("Module/Func Name");
							
							boolean keepGoing = true;
							
							DefaultMutableTreeNode nodeParent = (DefaultMutableTreeNode) node;
							
							while (keepGoing)
							{
								nodeParent = (DefaultMutableTreeNode) nodeParent.getParent();
								if (nodeParent.getUserObject() instanceof BlameFunction) 
									keepGoing = false;
							}
														
							BlameFunction bf = (BlameFunction) nodeParent.getUserObject();

							
							ExitSuper es = (ExitSuper) nodeInfo;
							Vector<String> sFramesEmpty = new Vector<String>();
							stackNameList.setListData(sFramesEmpty);


							SourceFile sFile = (SourceFile) SourceContainer.sourceFiles.get(bf.getModuleName());
							
							if (sFile == null)
								return;
							sourceBodyList.setListData( sFile.getLines() );

							//sourceBodyList.setCellRenderer(new MyCellRenderer());
							sourceBodyList.setSelectedIndex(es.lineNum-1);
							sourceBodyList.ensureIndexIsVisible(es.lineNum-1);

							localBT.metadataPane.setText(es.printDescription());

						}

						if (/*node.getLevel() == 4 || */node.getLevel() == 0 || node.getLevel() == 1)
						{
							stackFrameLabel.setText("");
							localBT.metadataPane.setText("");
							sourceBodyList.setListData(new Vector<String>());
							stackNameList.setListData(new Vector<String>());
						}
						
						if (node.getLevel() == 2)
						{
							
							stackFrameLabel.setText("Module/Func Name");
							BlameFunction bf = (BlameFunction) node.getUserObject();

							//ExitSuper es = (ExitSuper) nodeInfo;
							//Vector<String> sFramesEmpty = new Vector<String>();
							
							Vector<BlameFunction> bfVec = new Vector<BlameFunction>();
							bfVec.add(bf);
							stackNameList.setListData(bfVec);

							SourceFile sFile = (SourceFile) SourceContainer.sourceFiles.get(bf.getModuleName());
							
							if (sFile == null)
								return;
							
							sourceBodyList.setListData( sFile.getLines() );

							//sourceBodyList.setCellRenderer(new MyCellRenderer());
							
							sourceBodyList.setSelectedIndex(bf.getBeginNum());
							sourceBodyList.ensureIndexIsVisible(bf.getBeginNum());
							//sourceBodyList.setSelectedIndex(es.lineNum-1);
							//sourceBodyList.ensureIndexIsVisible(es.lineNum-1);

							//localBT.metadataPane.setText(es.printDescription());
						}
/*
						if (node.getLevel() == 5)
						{
							stackFrameLabel.setText("Stack Frames");
							VariableInstance vi = (VariableInstance)nodeInfo;
							Instance inst = vi.getInst();
							int index = inst.getInstanceNum();

							if (index < 0)
								inst = new Instance(0, "NULL");

							Vector<StackFrame> sFrames0 = inst.getStackFrames();

							if (sFrames0.size() > 0)
							{

								while (node.getLevel() != 2)
									node = (DefaultMutableTreeNode) node.getParent();

								BlameFunction bf = (BlameFunction)node.getUserObject();

								Iterator it = sFrames0.iterator();
								StackFrame sf = sFrames0.get(0);
								int sfIndex = 0;
								while (it.hasNext())
								{
									sf = (StackFrame)it.next();
									if (sf.getBf().equals(bf))
										break;
									sfIndex++;
								}


								//StackFrame sf = sFrames0.get(0);

								stackNameList.setSelectedIndex(sfIndex);
								stackNameList.setListData(sFrames0);


								SourceFile sFile = (SourceFile) SourceContainer.sourceFiles.get(sf.moduleName);
								sourceBodyList.setListData( sFile.getLines() );

								//sourceBodyList.setCellRenderer(new MyCellRenderer());
								sourceBodyList.setSelectedIndex(sf.lineNumber-1);
								sourceBodyList.ensureIndexIsVisible(sf.lineNumber-1);

							}
							else
							{
								sourceBodyList.setListData(new Vector<String>());
								stackNameList.setListData(new Vector<String>());
							}

							jMessages.revalidate();
							jMessages.repaint();

							sourceBody.revalidate();
							sourceBody.repaint();

						}
						*/


					}
				}
		);




		this.setSize(650,800);
		this.setLocation(410, 0);
		//setSize(650, 800);

	}

	private void addComponent( Component com, int row, int column,int width, int height)
	{
		// set gridx and gridy
		gbc.gridx = column;
		gbc.gridy = row;

		gbc.gridwidth = width;
		gbc.gridheight = height;

		// set constraints
		gbl.setConstraints(com, gbc);
		c.add(com);
	}

	

	private class ComplexCellRenderer extends JLabel implements ListCellRenderer {
		/**
		 * 
		 */
		private static final long serialVersionUID = 1L;
		
		//protected DefaultListCellRenderer defaultRenderer = new DefaultListCellRenderer();

		// We want to run this when we get to the proper tree level (with EV/EP)
		boolean atRightTreeLevel(DefaultMutableTreeNode node, int index)
		{	
			ExitSuper nodeInfo;
			
			if (node.getUserObject() instanceof ExitSuper)
				nodeInfo = (ExitSuper) node.getUserObject(); 
			else
				return false;
			
			if (nodeInfo.isField())
				return false;
			
			return nodeInfo.checkLineNum(index + 1);
		
		}
		
		boolean outsideFunction(int index)
		{
			if (whichNSIT == 0)
			{
				DefaultMutableTreeNode node = (DefaultMutableTreeNode)
				localBT.tree.getLastSelectedPathComponent();
				
				// We have the extra dialogue box out 
				if (localNSIT.isActive() && node.getLevel() >= 4)
				{
					StackFrame sf = (StackFrame) stackNameList.getSelectedValue();
					if (sf == null)
						return false;
					BlameFunction bf = sf.getBf();
					
					int beginLine = bf.getBeginNum();
					int endLine = bf.getEndNum();
					
					System.out.println("BF - " + bf.getName() + " " + beginLine + " " + endLine);
					
					if (index +1 < beginLine || index+1 > endLine )
						return true;

					return false;				
				}


				if (node.getLevel() == 0 || node.getLevel() == 1 )
				{
					return false;
				}
				else if (node.getLevel() == 2)
				{
					BlameFunction bf = (BlameFunction) node.getUserObject();

					int beginLine = bf.getBeginNum();
					int endLine = bf.getEndNum();
					if (index +1 < beginLine || index+1 > endLine )
						return true;

					return false;
				}
				else if (node.getLevel() == 3)
				{
					DefaultMutableTreeNode parentNode = (DefaultMutableTreeNode)  node.getParent();
					BlameFunction bf = (BlameFunction) parentNode.getUserObject();

					int beginLine = bf.getBeginNum();
					int endLine = bf.getEndNum();
					if (index +1 < beginLine || index+1 > endLine )
						return true;

					return false;
				}
				else if (node.getLevel() >= 4)
				{
					boolean keepGoing = true;

					DefaultMutableTreeNode nodeParent = (DefaultMutableTreeNode) node;

					while (keepGoing)
					{
						nodeParent = (DefaultMutableTreeNode) nodeParent.getParent();
						if (nodeParent.getUserObject() instanceof BlameFunction) 
							keepGoing = false;
					}


					BlameFunction bf = (BlameFunction) nodeParent.getUserObject();

					int beginLine = bf.getBeginNum();
					int endLine = bf.getEndNum();
					if (index +1 < beginLine || index+1 > endLine )
						return true;

					return false;

				}
			}
			else if (whichNSIT == 1)
			{
				return true;
			}
					
			return true;
		}

		boolean isVariableDecl(int index)
		{
			if (whichNSIT == 0)
			{
				DefaultMutableTreeNode node = (DefaultMutableTreeNode)
				localBT.tree.getLastSelectedPathComponent();

				if (node == null) return false;

				if (!(node.getUserObject() instanceof ExitSuper))
					return false;
			
				ExitSuper nodeInfo = (ExitSuper) node.getUserObject();
				if (nodeInfo.getLineNum() == index + 1)
					return true;
			}
			else if (whichNSIT == 1)
			{
				return false;
			}
			
			return false;
			
		}
		
		boolean indexRan(int index)
		{
			if (whichNSIT == 0)
			{
				DefaultMutableTreeNode node = (DefaultMutableTreeNode)
				localBT.tree.getLastSelectedPathComponent();

				return atRightTreeLevel(node, index);
			}
			else if (whichNSIT == 0)
			{
				return false;
			}
			
			return false;
		}

		public Component getListCellRendererComponent(JList list, Object value,
				int index, boolean isSelected, boolean cellHasFocus) {

			//System.out.println("Firing");
			setText(value.toString());
			this.setOpaque(true);
			setText(value.toString());
			if (isVariableDecl(index))
			{
				setBackground(isSelected ? Color.blue :  Color.cyan);
				setForeground(isSelected ? Color.white : Color.white);				
			}
			else if (indexRan(index))
			{
				setBackground(isSelected ? Color.pink : Color.red);
				setForeground(isSelected ? Color.white : Color.white);
			}
			else if (outsideFunction(index))
			{
				setBackground(isSelected ? Color.black : Color.white);				
				//setBackground(isSelected ? Color.lightGray : Color.white);
				//setForeground(isSelected ? Color.white : Color.lightGray);
				setForeground(isSelected ? Color.white : Color.black);
			}
			else
			{
				//setBackground(isSelected ? Color.lightGray : Color.white);
				setBackground(isSelected ? Color.black : Color.white);				
				setForeground(isSelected ? Color.white : Color.black);
			}
			return this;
		}
	}
}







/*
variableNameList.addListSelectionListener(
		new ListSelectionListener() {
			public void valueChanged( ListSelectionEvent e)
			{
				if ( e.getValueIsAdjusting() == false)
				{
					int index = variableNameList.getSelectedIndex();
					if (index >= 0)
					{
						ExitVariable ev = (ExitVariable) variableNameList.getSelectedValue();
						//localVMD._resultArea.setText(ev.stringEV());
						localVMD._resultArea.setText(ev.toString());
					}
				}
			}
		}
	);

programNameList.addListSelectionListener(
		new ListSelectionListener() {
			public void valueChanged( ListSelectionEvent e)
			{
				if ( e.getValueIsAdjusting() == false)
				{
					int index = programNameList.getSelectedIndex();
					if (index >= 0)
					{
						ExitProgram ep = (ExitProgram) programNameList.getSelectedValue();
						//localVMD._resultArea.setText(ep.stringEP());
						localVMD._resultArea.setText(ep.toString());
					}
				}
			}
		}
	);
 */





/*
bnw.instanceList.addListSelectionListener(
		new ListSelectionListener() {
			public void valueChanged( ListSelectionEvent e)
			{
				if ( e.getValueIsAdjusting() == false)
				{
//					Get the current stack frame


					int index = localBNW.instanceList.getSelectedIndex();
					Instance instance0 = (Instance) localBNW.instanceList.getSelectedValue();

					if (index < 0)
						instance0 = new Instance(0, "NULL");

					Vector<StackFrame> sFrames0 = instance0.getStackFrames();

					if (sFrames0.size() > 0)
					{
						StackFrame sf = sFrames0.get(0);

						stackNameList.setListData(sFrames0);


						// From stack frame get variable list
						BlameFunction bf = sf.bf;
						//variableNameList.setListData(bf.getExitVariables().values().toArray());
						//programNameList.setListData(bf.getExitPrograms().values().toArray());

						// From stack frame get Source file

						SourceFile sFile = (SourceFile) SourceContainer.sourceFiles.get(sf.moduleName);
						sourceBodyList.setListData( sFile.getLines() );

						//sourceBodyList.setCellRenderer(new MyCellRenderer());
						sourceBodyList.setSelectedIndex(sf.lineNumber-1);
						sourceBodyList.ensureIndexIsVisible(sf.lineNumber-1);

					}
					else
					{
						//variableNameList.setListData(new Vector<String>());
						//programNameList.setListData(new Vector<String>());
						sourceBodyList.setListData(new Vector<String>());
						stackNameList.setListData(new Vector<String>());
					}

					jMessages.revalidate();
					jMessages.repaint();

					sourceBody.revalidate();
					sourceBody.repaint();

					//jBody.revalidate();
					//jBody.repaint();

					//pBody.revalidate();
					//pBody.repaint();

				}
			}
		}
);
*/

/*
// Event Handler for Messages
variableNameList.addListSelectionListener(
		new ListSelectionListener() {
			public void valueChanged( ListSelectionEvent e)
			{
				if (variableNameList.getMaxSelectionIndex() == variableNameList.getMinSelectionIndex()
						&& e.getValueIsAdjusting() == false)
				{
					int index = stackNameList.getSelectedIndex();

					SourceFile sFile = (SourceFile) sourceContainer.getSourceFiles().get(index);

					sourceBodyList.setListData( sFile.getLines() );

					sourceBodyList.setCellRenderer(new MyCellRenderer());


				}
				else if (e.getValueIsAdjusting() == false)
				{

				}



				jBody.revalidate();
				jBody.repaint();
			}


		}
);*/		





/*
private class MyCellRenderer extends JLabel implements ListCellRenderer {
	Vector covered;

	public MyCellRenderer(Vector v) {
		setOpaque(true);
		covered = v;
	}

	public MyCellRenderer()
	{
		setOpaque(true);
	}

	public boolean indexCovered(int i)
	{
		for (int a = 0; a < covered.size(); a++)
		{
			if (i == ((Integer) covered.get(a) ).intValue()-1)
				return true;
		}
		return false;
	}

	public boolean indexRan(int i)
	{
		return false;
	}

	public Component getListCellRendererComponent(
			JList list,
			Object value,
			int index,
			boolean isSelected,
			boolean cellHasFocus)
	{

		setText(value.toString());
		if (indexRan(index)){
			setBackground(isSelected ? Color.pink : Color.red);
			setForeground(isSelected ? Color.white : Color.white);
		}
		else
		{
			setBackground(isSelected ? Color.lightGray : Color.white);
			setForeground(isSelected ? Color.white : Color.black);
		}
		return this;
	}
}*/



