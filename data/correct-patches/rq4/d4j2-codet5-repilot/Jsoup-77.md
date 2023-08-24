# Repilot Patch

```

        String elName = endTag.name().toLowerCase();
```

# Developer Patch

```
        String elName = endTag.normalName();
```

# Context

```
--- bug/Jsoup-77/src/main/java/org/jsoup/parser/XmlTreeBuilder.java

+++ fix/Jsoup-77/src/main/java/org/jsoup/parser/XmlTreeBuilder.java

@@ -114,7 +114,8 @@

      * @param endTag tag to close
      */
     private void popStackToClose(Token.EndTag endTag) {
-        String elName = endTag.name();
+
+        String elName = endTag.name().toLowerCase();
         Element firstFound = null;
 
         for (int pos = stack.size() -1; pos >= 0; pos--) {
```

# Note

protected String normalName; // lc version of tag name, for case insensitive tree build