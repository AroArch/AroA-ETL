qa_level_query = """
SELECT TOP (1000) 
      p.[strLName]
  FROM [ousarchiv].[ousarchiv].[PersData] as p
  JOIN [ousarchiv].[ousarchiv].[attributeQuality] as q
  on p.lObjId = q.lObjId and p.lCountId = q.lCountId
  where q.lAttTypeId = 1 and q.lSubTypeNo = 1 and q.strQLevel >= 4;
"""

persdata_query = """
SELECT schm.[strSchemaCode]
      ,schm.[lSchemaId]
      ,[lSequenceNo]
      ,[lPrevSeqNo]
      ,[lNextSeqNo]
      ,p.[lObjId]
      ,[strLName]
      ,[strLNamePhon]
      ,[lLNameType]
      ,[strGName]
      ,[strGNamePhon]
      ,[lGNamePos]
      ,[strDoB]
      ,[lDoBPos]
      ,[lNumber]
      ,[strNPref]
      ,[strCamp]
      ,[lPersId]
      ,[strSortValue]
      ,[lGapSize]
      ,[strSortValueAlph]
      ,[strSortValuePhon]
      ,[strPrisNo]
      ,p.[lCountId]
      ,[strNumber]
      ,[strValue_l1] as TDNumber
  FROM [ousarchiv].[ousarchiv].[PersData] as p
  LEFT JOIN (SELECT a.lObjId
                     , v.strValue_l1
		     , a.lCountId
        FROM [ousarchiv].[ousarchiv].[Attribute] as a  
        INNER JOIN [ousarchiv].[ousarchiv].[AttributeValue] as v  
		ON a.lValueId = v.lValueId AND a.lAttTypeId = v.lattTypeId AND a.lSubTypeNo = v.lSubTypeNo
        WHERE v.lAttTypeId = 1 and v.lSubTypeNo = 99 and v.strValue_l1 != '') as attr
  ON p.lObjId = attr.lObjId and p.lCountId = attr.lCountId
  LEFT JOIN [ousarchiv].[ousarchiv].[ArchiveSchema] as schm
  ON schm.lSchemaId = p.lSchemaId;
"""

bestand_query = lambda bestand_nr :  f"""
select bestand.lSchemaId,
       Pers.lObjId,
       Pers.lCountId,
       Pers.strLName,
       Pers.lLNameType,
       Pers.strGName,
       Pers.strDoB,
       Pers.lNumber as prisoner_number,
       pob.strPoB,
       tdnum.TD_number
from (select val.lObjId, schm.strSchemaCode as lSchemaId
      from ousarchiv.ValNodes as val
      join ousarchiv.ArchiveSchema schm
      on val.lSchemaId = schm.lSchemaId
      where schm.strSchemaCode in ('{bestand_nr}')) as bestand
join ousarchiv.PersData as Pers
on Pers.lObjId = bestand.lObjId
left join (select attr.lObjId as lObjId, attr.lCountId as lCountId, aVal.strValue_l1 as strPoB
      from ousarchiv.Attribute as attr
      join ousarchiv.AttributeValue as aVal
      on attr.lValueId = aVal.lValueId 
      where aVal.lAttTypeId = 1 and aVal.lSubTypeNo = 8) as pob
on Pers.lObjId = pob.lObjId and Pers.lCountId = pob.lCountId
left join (select attr.lObjId as lObjId, attr.lCountId as lCountId, aVal.strValue_l1 as TD_number
      from ousarchiv.Attribute as attr
      join ousarchiv.AttributeValue as aVal
      on attr.lValueId = aVal.lValueId 
      where aVal.lAttTypeId = 1 and aVal.lSubTypeNo = 99) as tdnum
on Pers.lObjId = tdnum.lObjId and Pers.lCountId = tdnum.lCountId;
"""
